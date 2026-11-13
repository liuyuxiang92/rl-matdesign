"""Real native HPO signals, synchronized restarts, and terminal exports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import textwrap
import time

import pytest
import torch
import yaml

from rl_matdesign.hpo import job_runner
from rl_matdesign.hpo.algorithm_profiles import initial_parameters
from rl_matdesign.hpo.controller import Campaign
from rl_matdesign.hpo.calibration import restore_quality_scale
from rl_matdesign.hpo.restart import _inspect_job, inspect_restart
from rl_matdesign.hpo.spec import load_plan
from tests.test_hpo_checkpoint_retention import _assert_nested_equal


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX cooperative signal integration')


def _plan(path, method, *, repeats=1, parallel=1):
    scenario = {
        'method': 'a2c' if method == 'a2c' else 'dqn',
        'properties': [{'name':'energy', 'predictor':'dummy', 'direction':'min', 'objective':'mean'}],
        'groups': [{'name':'metal','kind':'composition',
                    'species_set':['Ni','Fe','Co','Cr','Mn','Cu','Zn','Ti'],
                    'fraction_set':[f'{i/10:.1f}' for i in range(1,9)],'n_components':3}],
        'eval_seed':13, 'pg_warmup_eps':2, 'pg_batch_eps':2,
        'dqn_hidden_dim':16, 'dqn_buffer_size':100, 'dqn_grad_steps_per_ep':2,
        'gen_temperature':1., 'num_gen_eps':0, 'exploration_gen_eps':0,
    }
    parameters = ({'pg_batch_eps':2} if method == 'a2c' else
                  {'dqn_batch_size':2, 'dqn_lr':.001, 'dqn_eps_min':.2,
                   'warmup_budget_fraction':.2, 'epsilon_anneal_budget_ratio':1.})
    if method == 'dqn-bootstrap':
        parameters['dqn_target_update_freq'] = 2
    budget = {'mode':'absolute_unique','min_unique':12,'initial_unique':12,'max_unique':12}
    (path/'scenario.yaml').write_text(yaml.safe_dump(scenario, sort_keys=False))
    source = path/'hpo.yaml'
    source.write_text(yaml.safe_dump({
        'base_config':'scenario.yaml', 'method':method, 'random_seed':42, 'parallel_jobs':parallel,
        'evaluation':{'threshold':.5,'max_attempts_per_unique':1.5},
        'training':{'repeats':repeats,'budget':budget,'parameter_ranges':{
            name:{'type':'categorical','choices':[value]} for name,value in parameters.items()}},
        'generation':{'repeats':1,'budget':budget,'initial_values':[1.]},
    }, sort_keys=False))
    return load_plan(source), parameters


SIGNAL_PAID_SAMPLE = textwrap.dedent('''
    import os, runpy, signal
    from rl_matdesign.utils.candidate_records import CandidateRecorder
    original = CandidateRecorder.success
    sent = False
    def success(self, episode):
        global sent
        result = original(self, episode)
        if not sent and self.budget.unique_evaluated >= int(os.environ['STOP_AFTER_UNIQUE']):
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return result
    CandidateRecorder.success = success
    runpy.run_module('rl_matdesign.hpo.worker', run_name='__main__')
''')


class _WorkerHook:
    def __init__(self, code, extra_env):
        self.code, self.extra_env = code, extra_env

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, argv, **kwargs):
        assert argv[1:3] == ['-m','rl_matdesign.hpo.worker']
        return subprocess.Popen([argv[0],'-B','-c',self.code,*argv[3:]],
                                **{**kwargs,'env':dict(kwargs['env'],**self.extra_env)})


def _signalled(campaign, parameters, monkeypatch, stop_after, *, phase='train', model=None, seed=17):
    with monkeypatch.context() as patch:
        patch.setattr(job_runner,'subprocess',_WorkerHook(SIGNAL_PAID_SAMPLE,{'STOP_AFTER_UNIQUE':str(stop_after)}))
        return campaign.runner(parameters,12,seed,phase,model=model)


def _directory(campaign,result):
    return campaign.out/'jobs'/result['job_id']


def _completed(campaign,result):
    path = _directory(campaign,result)
    assert result['status']=='COMPLETED',(result,(path/'stderr.log').read_text())
    return path


def _checkpoint(path,name='checkpoint.pt'):
    return torch.load(path/name,map_location='cpu',weights_only=False)


def _single_checkpoint(path):
    assert (path/'checkpoint.pt').is_file() and not (path/'checkpoint.pt').is_symlink()
    assert not list(path.glob('checkpoint-ep*.pt'))
    assert not list(path.glob('checkpoint-iter*.pt'))


def _generation_campaign(plan, out, training_campaign):
    campaign = Campaign(plan,out)
    # These independent directories are test branches of the same trained
    # model and share its verified immutable scoring calibration.
    shutil.copyfile(training_campaign.out/'quality_calibration.json',
                    campaign.out/'quality_calibration.json')
    restore_quality_scale(campaign.plan,campaign.out)
    return campaign


@pytest.mark.parametrize('method',['a2c','dqn-mc','dqn-bootstrap'])
def test_paid_sample_signal_then_native_restart_matches_uninterrupted(tmp_path,monkeypatch,method):
    monkeypatch.chdir(tmp_path)
    plan,parameters = _plan(tmp_path,method)
    interrupted = Campaign(plan,tmp_path/'interrupted')
    # A2C stops after the first sample of a two-sample update; DQN stops
    # after its two warmup compositions and the first online composition.
    result = _signalled(interrupted,parameters,monkeypatch,1 if method=='a2c' else 3)
    path = _directory(interrupted,result)
    assert result['returncode']==128+signal.SIGTERM,(result,(path/'stderr.log').read_text())
    assert _inspect_job(path,method)['classification']=='SYNCHRONIZED'
    before_records = (path/'candidate_records.jsonl').read_bytes()
    before_budget = json.loads((path/'candidate_budget.json').read_text())
    saved = _checkpoint(path)
    assert before_budget['status']=='INTERRUPTED'
    assert 0 < before_budget['unique_evaluated'] < 12
    assert before_budget['attempts_started']==saved['hpo_state']['budget']['attempts_started']
    if method=='a2c':
        assert saved['iteration']==1 and saved['episodes_completed']==2
    _single_checkpoint(path)

    resumed = interrupted.runner(parameters,12,17,'train')
    _completed(interrupted,resumed)
    assert (path/'candidate_records.jsonl').read_bytes().startswith(before_records)
    control = Campaign(plan,tmp_path/'control')
    control_path = _completed(control,control.runner(parameters,12,17,'train'))
    actual,expected = _checkpoint(path),_checkpoint(control_path)
    keys = (['policy_state','value_net_state','opt_actor_state','opt_critic_state','iteration']
            if method=='a2c' else ['qnet_state','target_net_state','opt_state','buffer','eps','training_episodes_completed'])
    for key in keys+['rng_state','training_budget_state','hpo_state','episodes_completed']:
        _assert_nested_equal(actual[key],expected[key])
    assert (path/'candidate_records.jsonl').read_bytes()==(control_path/'candidate_records.jsonl').read_bytes()
    assert json.loads((path/'candidate_budget.json').read_text())==json.loads((control_path/'candidate_budget.json').read_text())
    _single_checkpoint(path)


@pytest.mark.parametrize('method',['a2c','dqn-mc','dqn-bootstrap'])
def test_generation_signal_native_restart_and_terminal_export(tmp_path,monkeypatch,method):
    monkeypatch.chdir(tmp_path)
    plan,parameters = _plan(tmp_path,method)
    training_campaign = Campaign(plan,tmp_path/'training')
    model = training_campaign.runner(parameters,12,17,'train')
    _completed(training_campaign,model)
    generation_parameters = {**parameters,'gen_temperature':1.}
    control = _generation_campaign(plan,tmp_path/'generation-control',training_campaign)
    control_path = _completed(control,control.runner(generation_parameters,12,101,'generation',model=model))
    expected = _checkpoint(control_path,'generation_checkpoint.pt')

    interrupted = _generation_campaign(plan,tmp_path/'generation-interrupted',training_campaign)
    result = _signalled(interrupted,generation_parameters,monkeypatch,1,phase='generation',model=model,seed=101)
    path = _directory(interrupted,result)
    assert result['returncode']==128+signal.SIGTERM,(result,(path/'stderr.log').read_text())
    assert _inspect_job(path,method)['classification']=='SYNCHRONIZED'
    prefix = (path/'candidate_records.jsonl').read_bytes()
    paid = json.loads((path/'candidate_budget.json').read_text())
    saved = _checkpoint(path,'generation_checkpoint.pt')
    assert paid['attempts_started']==saved['hpo_state']['budget']['attempts_started']
    assert paid['unique_evaluated']==1
    _completed(interrupted,interrupted.runner(generation_parameters,12,101,'generation',model=model))
    actual = _checkpoint(path,'generation_checkpoint.pt')
    for key in ['rng_state','hpo_state']:
        _assert_nested_equal(actual[key],expected[key])
    assert (path/'candidate_records.jsonl').read_bytes().startswith(prefix)
    assert (path/'candidate_records.jsonl').read_bytes()==(control_path/'candidate_records.jsonl').read_bytes()

    # SIGTERM at the final unique candidate must not stop between its final
    # checkpoint and export: the terminal job finishes and is reusable CACHED.
    terminal = _generation_campaign(plan,tmp_path/'generation-terminal',training_campaign)
    final = _signalled(terminal,generation_parameters,monkeypatch,12,phase='generation',model=model,seed=101)
    terminal_path = _completed(terminal,final)
    assert final['returncode']==0
    assert _inspect_job(terminal_path,method)['classification']=='CACHED'
    assert (terminal_path/'generated.csv').is_file()
    assert (terminal_path/'candidate_records.jsonl').read_bytes()==(control_path/'candidate_records.jsonl').read_bytes()


COORDINATOR = textwrap.dedent('''
    import os, subprocess, sys
    from rl_matdesign.hpo import job_runner
    code = os.environ['TEST_WORKER_HOOK']
    class Proxy:
        def __getattr__(self,name):
            return getattr(subprocess,name)
        def Popen(self,args,**kwargs):
            return subprocess.Popen([args[0],'-B','-c',code,*args[3:]],**kwargs)
    job_runner.subprocess = Proxy()
    from rl_matdesign.cli import main
    main()
''')

WAIT_FOR_PARENT_STOP = textwrap.dedent('''
    import json, os, runpy, time
    from pathlib import Path
    from rl_matdesign.utils import hpo_stop
    from rl_matdesign.utils.candidate_records import CandidateRecorder
    original = CandidateRecorder.success
    waiting = False
    def success(self,episode):
        global waiting
        result = original(self,episode)
        if not waiting:
            waiting = True
            marker = Path(os.environ['TEST_MARKER_DIRECTORY']) / (str(os.getpid())+'.json')
            marker.write_text(json.dumps({'pid':os.getpid(),'job':str(self.path)}))
            deadline = time.monotonic()+30
            while hpo_stop._active_stop.signum is None:
                if time.monotonic()>deadline:
                    raise RuntimeError('Test coordinator did not propagate its stop signal')
                time.sleep(.01)
        return result
    CandidateRecorder.success = success
    runpy.run_module('rl_matdesign.hpo.worker',run_name='__main__')
''')


def test_parallel_coordinator_sigterm_waits_for_safe_workers_and_native_restart(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan,_ = _plan(tmp_path,'a2c',repeats=2,parallel=2)
    source = tmp_path/'hpo.yaml'
    raw = yaml.safe_load(source.read_text())
    raw['training']['parameter_ranges'] = {
        name:{'type':'categorical','choices':[value]}
        for name,value in initial_parameters(plan).items()}
    source.write_text(yaml.safe_dump(raw,sort_keys=False))
    out = tmp_path/'campaign'
    markers = tmp_path/'worker-markers'
    markers.mkdir()
    environment = dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                       TEST_WORKER_HOOK=WAIT_FOR_PARENT_STOP,TEST_MARKER_DIRECTORY=str(markers))
    process = subprocess.Popen([sys.executable,'-B','-c',COORDINATOR,'hpo','run',
                                '--hpo-config',str(tmp_path/'hpo.yaml'),'--out',str(out),'--stage','ranges'],
                               cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    pids=[]
    try:
        deadline=time.monotonic()+60
        while len(list(markers.glob('*.json')))<2:
            assert process.poll() is None,process.communicate(timeout=5)
            assert time.monotonic()<deadline,'Parallel workers did not both reach paid samples'
            time.sleep(.05)
        pids=[json.loads(path.read_text())['pid'] for path in markers.glob('*.json')]
        process.send_signal(signal.SIGTERM)
        stdout,stderr=process.communicate(timeout=60)
        assert process.returncode==130,(stdout,stderr)
        state=json.loads((out/'state.json').read_text())
        assert state['status']=='INTERRUPTED',state
        report=inspect_restart(out)
        assert report['ready'] and report['job_counts']['SYNCHRONIZED']==2,report
        preserved={Path(row['jobdir']):(Path(row['jobdir'])/'candidate_records.jsonl').read_bytes()
                   for row in report['jobs']}
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid,0)
        resumed=subprocess.run([sys.executable,'-B','-c','from rl_matdesign.cli import main; main()',
                                'hpo','restart','--out',str(out)],cwd=ROOT,env=environment,
                               capture_output=True,text=True,timeout=120)
        assert resumed.returncode==0,(resumed.stdout,resumed.stderr)
        assert json.loads((out/'state.json').read_text())['status']=='COMPLETED'
        for path,prefix in preserved.items():
            assert (path/'candidate_records.jsonl').read_bytes().startswith(prefix)
            assert _inspect_job(path,'a2c')['classification']=='CACHED'
            _single_checkpoint(path)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        for path in markers.glob('*.json'):
            pid=json.loads(path.read_text())['pid']
            try:
                os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
