#!/usr/bin/env python3
"""Read-only source evidence for the DQN performance update; no model imports.

Run with the same Python environment as rl-matdesign BEFORE updating the package:
  python hpo_source_evidence.py --runs-dir ./runs
The script prints JSON to stdout and does not modify campaigns or source files.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

BASELINE_REVISION = "ba6bc1cf3dec6e050d44fd278e1e41243f54dbca"
BASELINE_FILES = {'abcde_ooh/__init__.py': '488169f6489d96db9cf75c4fb1e4ed7c6eb6e7ad40cdb124bd00463a9443df52', 'abcde_ooh/constraints/__init__.py': '070992ecbf9545c395575bebee855ecdcb59a0e28d952766a442cd1f32eb9dbb', 'abcde_ooh/constraints/phase_sampler.py': '614fa5fd827efe97e13857008ae3a4654cfe1fa6d75e7dc63016fd99c59f7a3e', 'abcde_ooh/constraints/primary_phase.py': 'bc0492543d1414c0a0abed554fbcbd9b9a54d67dc6321fb5dcc9600eddc4be30', 'abcde_ooh/dp_predictor.py': '42f391732865a8cdf3e0da24472bd604199e9ec03357aea75ed28cf1f3688cf3', 'abcde_ooh/encoding.py': 'c73ae494ed4cbf8e7c5f8f18c24e1fc8997e8f1f0ba79cae5eda66a26764ec45', 'abcde_ooh/env.py': '9d02886b5ca472373870fe3c1dfb28541e785a90858e208e5b65d9015cb21981', 'rl_matdesign/__init__.py': 'aee1fa833ca7eff533837738aeb6442e3a553f864bb986144bebaf73e8b948ca', 'rl_matdesign/action_encoding.py': '96d9ac74f3c2697b619c70dbe166a62fef6f5a86c5ea07d2b9aeac7ec3930745', 'rl_matdesign/action_features.py': '74247862e90fe052cfbfdba47f8380ca81df57132a2174742848f8b9e8fea913', 'rl_matdesign/cli.py': '85c3a9e29b368dd22435aa68ab3e043b4d57b3337873b8d0438c65ceefce6d11', 'rl_matdesign/config_schema.py': '4859366d8981675e22e8e2f18905725a632530dce129f5c789743d5a652125ae', 'rl_matdesign/configuration.py': '088f0d996f1beb5750fe4c60c5d49ba77ff7e0b7574a953a06fc42c4aa3407a6', 'rl_matdesign/constraints/__init__.py': '8ca31cdbe6029d5ba4549691405743cae35358060a31ea75d91ae9596781f057', 'rl_matdesign/constraints/base.py': '76529ed0591ad549ebb03748c339463ddbb3a246f833f9940e9bff5eaefc854e', 'rl_matdesign/constraints/chain.py': 'e8c44cacc88daaf9f9c4cfaa2f6a623004e4d57e25b5357ede0617078b143ecd', 'rl_matdesign/constraints/charge.py': 'd295c60e788c1e6c5ecc13dd5db4ef57b443a6b4d2428d08147aae8d3e772285', 'rl_matdesign/constraints/last_step_element.py': 'a9929f14abc4955d64cdb705413dc93c1ca8e4ecd84d9d967fc87811ca3982a7', 'rl_matdesign/constraints/phase_pattern.py': '3376654a1022646467d74f35c213d2d88758a11eccebf2a457368a9c33483660', 'rl_matdesign/constraints/smact_filter.py': '64ac50b6ea156dc4ed3981acf6f46854e38786877a76f63bf7f8afc3b99243d3', 'rl_matdesign/constraints/sse_doping.py': '8affcd7f97469d9d0ece7f7072236acb3be5325248e2cd57235a61d686659048', 'rl_matdesign/encoding.py': 'ddd53843b9a41ca76598a67f5e363e08075995666f2c4857213cd45baeaccaeb', 'rl_matdesign/env.py': 'afb90e11b30f1ad50399b91040428168bc4bf4e585d44b59462fdf136453475d', 'rl_matdesign/env_integer.py': '4dab938718041113b73d1f2b2c962d563fb3d7fa94db5a5bb648f0ad74cbd07b', 'rl_matdesign/env_multigroup.py': 'e483f8dc76dfdd98ff58a929a62ee010eb9cf43bf3e5b8e54a731e7901986815', 'rl_matdesign/experiment.py': '35e480ebb62b9e278b68cfc7ea078ba886d4b3d3dc57e5d1c43dcde5b3f09571', 'rl_matdesign/featurization.py': '9229f56b01c31c696ae670cbff0346b9040c1c372d799fa8e42f221381d77832', 'rl_matdesign/hpo/__init__.py': '93285a9c9ceacca008460e51b5a68ae7edd3a6e67209c54a1b457bb5ff925b41', 'rl_matdesign/hpo/__main__.py': '935a1c1166b0c1ea35a82256345000bf2c73ded718d77773bc27a71ecce28f7d', 'rl_matdesign/hpo/algorithm_profiles.py': 'af463faf1a0628c3b90917aa8c1fe7f6f7dfa6a080331268178a94f665f66243', 'rl_matdesign/hpo/artifacts.py': 'b4694603e73216cc729dded2ea0eec1cd48c64428bc9ac6893bf85b1d176389a', 'rl_matdesign/hpo/budget_search.py': '1948c721b418dd90949963b30f2fb226013af5f30600528e1a01e0d5c4b25334', 'rl_matdesign/hpo/calibration.py': '50f3da5cef130bbb9c5a9267b864eb8b48627655136be7b35d50793bcd665cd6', 'rl_matdesign/hpo/cli.py': '007b6410adfb84c091ebe1d454bc286a5c5f8f38156386bcefa34ef7b4053b30', 'rl_matdesign/hpo/controller.py': '97891ef82921b862c567cc698c2e8d2fecef66b7793da4605d36218ff214a14a', 'rl_matdesign/hpo/domain.py': 'a0efe48e09e6d827437a830744c0ff9929b3aded14fdcb25e3dbc582a0bb8eb1', 'rl_matdesign/hpo/job_runner.py': '05278602cc1f3c5ece578198190b974e529fd3184f7383f91cea0710d334103d', 'rl_matdesign/hpo/metric.py': 'a950cb89bc28eafd6370155f4db0c5e38ccdd449b2f31c9ed42085600f340bc6', 'rl_matdesign/hpo/range_search.py': 'cc9b938c5ade46051adfd768e3ebcf4413411738ddc59642aceae37ecba33478', 'rl_matdesign/hpo/reporting.py': '9f40fbb82320ff705a5fc6204e9b97e1e6ca23a8451a844f6d3b2c299cadacca', 'rl_matdesign/hpo/restart.py': 'f7c73ca78eb1064744952507be979371da9545af6f610463dff0b5ebbaf8eabf', 'rl_matdesign/hpo/runner.py': 'a813705a2e527ab0291082542002ecf5723036bb0f7a61caf8e5e8973e8e4ebe', 'rl_matdesign/hpo/scheduling.py': '22a7c8dce6b2e4e9c4505245fd5e3992af558e8142987fab789616833fcb2048', 'rl_matdesign/hpo/scoring.py': '189a16338ac4f8b9d6900c6cc5f83036d4f848ccacd01a556e6df60e458cd1fb', 'rl_matdesign/hpo/search_space.py': '6005b91d80c379d7a656ae5085dc9b7e29896778bc56dbab563a4c5f56f50743', 'rl_matdesign/hpo/spec.py': '831b7cb242e0c10713973bf9ccd9b9eb4ee85bfa788dff2b16732f9b739de688', 'rl_matdesign/hpo/spec_v2.py': '2c9ca68ee905ae84e85ffba9f42b94f8eaa1a0d05926e3f4c40234568afd7d0d', 'rl_matdesign/hpo/worker.py': '466407d26f92e37d3b29c2bc3e8c5d1a578349196186186b2795318b2767c336', 'rl_matdesign/model.py': '15bae9f19851bfad301d10c856bf9c2745b6157210a441f8bbbccca6b38cc7ba', 'rl_matdesign/predictors/__init__.py': '5fc4b43b472bbbfd9a02b2b4f47d8e6b2fa4662126af8a055546a7112479cb1a', 'rl_matdesign/predictors/base.py': '21ae3baeea114600b32544b74b15278a2a89fb71da48e771d1cd9bbe6348175c', 'rl_matdesign/predictors/builders/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'rl_matdesign/predictors/builders/defect_site.py': '774b0759ebc95f26e65d331fee10a82a21874b639fc0fa31add2dd3a14e1df2f', 'rl_matdesign/predictors/builders/site_pick.py': '0f3b7ee9f485e11ab27623fe655a0a83e7311240e23a83c20d14d4d885a4ca5f', 'rl_matdesign/predictors/builders/sse.py': 'ae2ec1d830a72748b1aea485d36eaf371c2e08c0e30991320733e3f9af8ebd50', 'rl_matdesign/predictors/builders/substitute.py': 'a84f11648cb73625994885b6a8d2c4adb24c25aabf336597fa6aa880ca131d0e', 'rl_matdesign/predictors/glassnet.py': 'fdd5cae0b3fef7dd111e0dbca462cd1f61268ab7b35ef0089a97ea599d3458a4', 'rl_matdesign/predictors/lookup.py': '9fd8455aba6575746377a1f0495084324c540ac864a1af9f6886b5874b1592b2', 'rl_matdesign/predictors/mgtransformer.py': '16c5801da02cd24efa504f8e2109b5438854681e32f0b7fab5197d9694308ec2', 'rl_matdesign/predictors/ooh.py': '3e543ca628668399fef4cc9360dcbe8a8fd4b043795d287f17d6186cd737b6f1', 'rl_matdesign/predictors/outputs.py': '0e236854f29cb9b57aa7e3be025f75582d1c280cba8c89931105e3dcb902cfa7', 'rl_matdesign/predictors/rf_magpie.py': '91e1065b54301acbb352f9eb08dda89af31ce9e02087663c36faf23f2cbacc3b', 'rl_matdesign/predictors/structure_score.py': '2dd3ec20dc23cf85ee1459718d5de6586a3555f9a9bc6c3f40c8d87a53c66d42', 'rl_matdesign/registry.py': 'ec8f7478dd9cf01ba5f74403d3c202551215f8f288051f30b6e9086ae52a3bfb', 'rl_matdesign/rewards.py': '6db4b40aa58a901c0e0daac6af1ff08d8db91e6702293a477308674ad1bffe64', 'rl_matdesign/sequence.py': 'a45574087d115f20a4fcd85d9a89db3e70d8d47758d0c9b67ab2bfa4b579adbe', 'rl_matdesign/structure_sources.py': '4a4543b58586def10e75360380c4fa00c3ba0d767e0265f269aee543376ffead', 'rl_matdesign/training.py': '80a2204967c9090243c2078faf9171d7995fa5aebfd18fb8f85246d0957cbd81', 'rl_matdesign/utils/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'rl_matdesign/utils/candidate_budget.py': '984295185971009fe3c3420d618e1036ac63462b5a72a1e6e2d91114e9e20659', 'rl_matdesign/utils/candidate_records.py': '8d455a6fad009700e875308f2898255ec083915cc9316794109abf97d7af4aa7', 'rl_matdesign/utils/composition.py': 'fd2d805b60d1ab5cac65c5d88fbb0c4c2f52b6eabce09dee50619c03148a029b', 'rl_matdesign/utils/composition_domain.py': 'a9390da17bbc2be2de36e9292ac44f91120a8865047cd673b0bcf97ba51d635d', 'rl_matdesign/utils/design_count.py': 'd7a14b63ad8baff7a1824f30caae5043b8c9811ad9298cf50626e20b52f5b81e', 'rl_matdesign/utils/dp_eval.py': 'e2186fc437d3d8fa02c6136a2cebc42347827dca38871848179b677a8e10f669', 'rl_matdesign/utils/episode_batch.py': '225f4a9c4cc61978db832356c4f655d44464c47ba7c9365407efca19f6194e7c', 'rl_matdesign/utils/execution_contract.py': '27a81aefe06ecf169d2d040d4610691316da9d785576c7b59c674cf5fb1528a9', 'rl_matdesign/utils/generation_resume.py': '574d1dc78bd4e701c550030407674a6fb79d8b168fccacb9afc3519da6851981', 'rl_matdesign/utils/hpo_stop.py': 'a940e33c7354abe81a522696386485980167347e8047170ce6b949da8ec124cc', 'rl_matdesign/utils/metrics.py': '94f21dc07bca9e9843a7e0c815fb6b92da19d36b916cd8809286066cf7d01059', 'rl_matdesign/utils/model_contract.py': '9ca348794368b1e8b6e44b18284a72a2d53ff1085da85bcdde716a99b96a91eb', 'rl_matdesign/utils/parallel_structures.py': '39d5ea5744d94f2a7b34c9441ae28a15fae9bebf60fd2bf63e786eac186c1e33', 'rl_matdesign/utils/prediction_records.py': '975e23739e4e7e8a8fd4bfdc0616aa61b0515b6b75c88438e6d604802544c010', 'rl_matdesign/utils/provenance.py': '5993d2a525f0b27d74350996e372fee5b2d2f05c26113522b039270a69b087bb', 'rl_matdesign/utils/reward_payload.py': 'c2d0d494b63588f39486743cc5a9d15b97f98bac8be9a9ee182d834ffb9229c5', 'rl_matdesign/utils/rollout_errors.py': '07d0b2871bef91edf8bd0a96757a7a654b8b1c5e29b3484adbe346e4d80a6f1c', 'rl_matdesign/utils/seeding.py': '662b04feefd581103531790cac6cea54ecceccdde082c8daa02733a2c38ad0ec', 'rl_matdesign/utils/stage_timing.py': '5ac3de4116aa15497cf115d2cf78116eb9414db224a02180e76c669c2c36c849', 'rl_matdesign/utils/structure.py': 'c317420453f0c382bda00078001a6082358785cfd685a56f144d947cdcfd5c0c', 'rl_matdesign/utils/structure_samples.py': 'be4fcf1fdd5077bff791cec64cefc1dc09891f2d44b55d55094988c80b5561d7', 'rl_matdesign/utils/structure_worker.py': '8b35d0f7e24ed413ba591a676d1855758825e5ad19ff6b474be050baf80d20ba', 'rl_matdesign/utils/timing.py': '1199d815ae946dd06f05d6d984a3901eb1584639f76b306067cb26370611c582', 'rl_matdesign/utils/training_budget.py': '3a84852943f28858cda25183c8f8ac0ef2ceb6a79b4a447542cb0d0b94cb4f85', 'rl_matdesign/utils/training_samples.py': '565b8da36444f3528fc5dc2898b7df718445359c9137c4c3a7f5fdb22ac919fa'}

def digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def hashes(inventory):
    h = hashlib.sha256()
    for name in sorted(inventory, key=Path):
        h.update(name.encode()); h.update(bytes.fromhex(inventory[name]))
    return {"execution_source_sha256": h.hexdigest(), "package_digest": digest({
        name.removeprefix("rl_matdesign/"): value for name, value in inventory.items()
        if name.startswith("rl_matdesign/")})}

def git(directory, *args):
    try:
        result = subprocess.run(["git", "-C", str(directory), *args],
            capture_output=True, text=True, timeout=15, check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--out", help="One existing campaign output directory")
    where.add_argument("--runs-dir", help="Directory whose immediate children are campaigns")
    args = parser.parse_args()
    spec = importlib.util.find_spec("rl_matdesign")
    if spec is None or spec.origin is None:
        raise SystemExit("rl_matdesign is not installed in this Python environment")
    package = Path(spec.origin).resolve().parent
    root = package.parent
    inventory = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob("*.py"))}
    differences = [{"path": name, "baseline": BASELINE_FILES.get(name),
                    "installed": inventory.get(name)}
                   for name in sorted(set(BASELINE_FILES) | set(inventory))
                   if BASELINE_FILES.get(name) != inventory.get(name)]
    git_root = git(root, "rev-parse", "--show-toplevel")
    git_data = {"root": git_root, "commit": git(root, "rev-parse", "HEAD"),
                "tracked_status": git(root, "status", "--porcelain", "--untracked-files=no")}
    repository = Path(git_root.get("stdout", ""))
    if git_root.get("returncode") == 0:
        try:
            source_scope = str(root.relative_to(repository))
        except ValueError:
            source_scope = "."
        for key, extra in (("untracked_python", []), ("ignored_python", ["--ignored"])):
            result = git(repository, "ls-files", "--others", "--exclude-standard", *extra,
                         "--", source_scope)
            result["stdout"] = [name for name in result.get("stdout", "").splitlines()
                                if name.endswith(".py")]
            git_data[key] = result
    if args.out:
        candidates = [Path(args.out)]
    elif args.runs_dir:
        candidates = sorted(p for p in Path(args.runs_dir).iterdir() if p.is_dir())
    else:
        candidates = []
    campaigns = []
    for directory in candidates:
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            saved = json.loads(manifest.read_text())
            original = {key: saved.get(key) for key in
                        ("plan", "domain", "software_digest", "assets", "configuration_sources")}
            campaigns.append({"out": str(directory.resolve()),
                "software_digest": saved.get("software_digest"),
                "contract_digest": saved.get("contract_digest"),
                "manifest_integrity": digest(original) == saved.get("contract_digest"),
                "method": saved.get("plan", {}).get("method")})
        except (OSError, ValueError, TypeError) as exc:
            campaigns.append({"out": str(directory.resolve()), "error": str(exc)})
    print(json.dumps({"read_only": True, "python": sys.executable,
        "installed_package": str(package), "source_root": str(root),
        "baseline_revision": BASELINE_REVISION, "baseline": hashes(BASELINE_FILES),
        "installed": hashes(inventory), "source_differences": differences,
        "source_files": inventory, "git": git_data, "campaigns": campaigns},
        indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
