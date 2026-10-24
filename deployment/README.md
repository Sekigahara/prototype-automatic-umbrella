# Deployment

Everything needed to run the umbrella outside the notebooks: test the Shelly and motor, replay held-out weather
through the trained model, and prepare that data. The device needs only this folder, not the training code.

| Script | What it does | Needs hardware? |
|---|---|---|
| `test_shelly.py` | checks the connection to the Shelly 2PM Gen4 and nudges the motor | Shelly + motor, or the simulator |
| `run_simulation.py` | replays recorded weather through the model, printing every input and decision | no |
| `extract_evaluation.py` | extracts held-out weather clips (never used for training) for the replay | no |
| `modules/fake_shelly.py` | a simulated Shelly + motor, so `test_shelly.py` runs without hardware | no |

**All commands below are run from this folder** (`cd deployment`), with the environment from
[Setup](#setup-linux-python-39) activated. Every option has a default except
`test_shelly.py --host`; `python <script> --help` lists them all.

## Quick start

```bash
cd deployment

# 1. replay held-out weather through the model (no hardware)
python run_simulation.py --limit 2 --every 30

# 2. try the motor control without hardware: simulator in terminal 1, test in terminal 2
python modules/fake_shelly.py
python test_shelly.py --host 127.0.0.1:8080 --move open --back

# 3. the real Shelly: read-only check first (never moves the motor)
python test_shelly.py --host 192.168.1.50
```

**To stop anything: press Ctrl+C.** It is always safe; see [Stopping safely](#stopping-safely).

## Setup (Linux, Python 3.9)

`requirements.txt` pins the direct dependencies; all of them install from binary wheels on Linux x86_64.
Python 3.9 is required. Use **either** a venv (option A) **or** conda (option B), not both.

### Option A: venv

Ubuntu 22.04 ships 3.10, so install 3.9 from the deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.9 python3.9-venv
```

Then, from the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r deployment/requirements.txt
python -m ipykernel install --user --name weather_env --display-name "Python (weather_env)"
```

Activate it again in every new shell with `source .venv/bin/activate`.

### Option B: Miniconda (own environment, any Python version)

Conda installs Python 3.9 itself, so no system Python or PPA is needed. Everything goes into its own environment,
here called **`umbrella`**; `base` is never touched. Use any name except `base`, and the same name in every step.
The installer steps follow Anaconda's
[Miniconda Linux install guide](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install),
for **aarch64** (ARM64, the Jetson Orin). On an x86_64 PC, replace `aarch64` with `x86_64` in steps 2–4.

1. **Check the architecture** and that conda is not installed yet:
   ```bash
   uname -m             # aarch64 on the Jetson Orin
   conda --version      # "command not found" -> install it (steps 2-4); a version -> skip to step 5
   ```
   Do not install a second copy over an existing one: if `~/miniconda3` already exists, the installer stops with
   "File or directory already exists". Never force it with `-u` or `-f`; that can break the installation.
2. **Download the aarch64 installer** into your home folder:
   ```bash
   cd ~
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
   ```
3. **Verify it** (recommended): compare the hash with the one listed for `Miniconda3-latest-Linux-aarch64.sh`
   at <https://repo.anaconda.com/miniconda/>. Only continue if they match.
   ```bash
   sha256sum ~/Miniconda3-latest-Linux-aarch64.sh
   ```
4. **Install:**
   ```bash
   bash ~/Miniconda3-latest-Linux-aarch64.sh
   ```
   Answer the prompts in this order:
   - press **Return** to read the licence, then type **`yes`** to accept it;
   - press **Return** to install to the default `/home/<user>/miniconda3`;
   - type **`yes`** to initialize conda, so the `conda` command works in new shells.

   It ends with "Thank you for installing Miniconda3!". Then reload the shell and check:
   ```bash
   source ~/.bashrc     # or close and reopen the terminal
   conda list           # "# packages in environment at /home/<user>/miniconda3:" and a package list
   ```
   If `conda` is still not found: run `~/miniconda3/bin/conda init`, then `source ~/.bashrc` again.
5. **Keep `base` out of the way.** After installing, every new shell starts in `base` (the prompt shows `(base)`),
   so a stray `pip install` would land there. Turn that off; `conda` keeps working, and `conda activate` still
   enters `base` when you really want it:
   ```bash
   conda config --set auto_activate_base false
   conda config --describe auto_activate_base    # shows: auto_activate_base: false
   source ~/.bashrc                               # the prompt no longer starts with (base)
   ```
6. **Create the environment with Python 3.9** (change `python=3.9` here for another version).
   It uses the conda-forge channel only (`--override-channels`), which has Python 3.9 builds for aarch64:
   ```bash
   conda create -n umbrella -c conda-forge --override-channels python=3.9 pip -y
   ```
7. **Isolate it from user-level packages.** Packages in `~/.local` (from `pip install --user`) would otherwise
   leak into the environment and mix versions:
   ```bash
   conda env config vars set -n umbrella PYTHONNOUSERSITE=1
   ```
8. **Activate it and check you are really inside it** before installing anything:
   ```bash
   conda activate umbrella
   which python         # must be ~/miniconda3/envs/umbrella/bin/python, not ~/miniconda3/bin/python (base)
   python --version     # Python 3.9.x
   python -c "import platform; print(platform.machine())"   # aarch64
   ```
9. **Install the requirements with the environment's own pip**, from the repository root.
   `python -m pip` (not plain `pip`) guarantees it installs into `umbrella`:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r deployment/requirements.txt
   python -m pip check                        # "No broken requirements found."
   ```
10. **Optional, for the notebooks:** register it as a Jupyter kernel.
   ```bash
   python -m ipykernel install --user --name umbrella --display-name "Python 3.9 (umbrella)"
   ```

Run `conda activate umbrella` in every new shell before using the scripts.

Rules that keep it clean:
- Install packages only with `python -m pip` inside the activated environment. Do not also `conda install` the
  same packages, or conda and pip will each manage their own copy.
- On the Jetson (aarch64), if `pip` finds no prebuilt wheel for a package and tries to compile it, install that one
  package from conda-forge first and rerun step 9:
  `conda install -n umbrella -c conda-forge --override-channels <package>=<version from requirements.txt>`.
- Keep `-c conda-forge --override-channels` on every `conda create` / `conda install`, as above, so the
  environment only uses conda-forge packages.
- Start over at any time, without affecting `base` or other environments:
  ```bash
  conda deactivate
  conda env remove -n umbrella
  ```
- Check which environments exist and which one is active (marked `*`): `conda env list`.

Miniconda on aarch64, from Anaconda's guide:
- The `linux-aarch64` builds target server-class ARM CPUs (Neoverse N1/N2) and may not run on some boards,
  Raspberry Pi in particular. If `conda` or `python` stops with "Illegal instruction", Miniconda does not support
  that CPU: use option A (venv) instead.
- Behind a proxy or firewall, `https://repo.anaconda.com` and `https://anaconda.org` must be reachable for the
  installer and for conda; `https://conda.anaconda.org` is needed for the conda-forge packages in step 6.
- A broken Miniconda can be repaired without losing environments: rename `~/miniconda3` to `~/miniconda_old`,
  run the installer again into `~/miniconda3`, copy the old contents back with
  `rsync -a ~/miniconda_old/ ~/miniconda3/`, check with `conda info --envs`, then delete `~/miniconda_old`.

`dataset/` (raw KMA files) is git-ignored, so copy it over separately if you need to re-run `simulation.ipynb`.
`simulated_dataset/`, `labeled_dataset/` and `models/umbrella_hmm.joblib` come with the repository.

---

## Workflow 1: try the motor control without hardware

The simulator behaves like the Shelly 2PM Gen4 with a motor: position (0 % closed to 100 % open), travel time,
limit switches at both ends, power drawn while turning, optional password. It prints every command and every motor start/stop.

1. **Terminal 1: start the simulator**
   ```bash
   python modules/fake_shelly.py                      # http://127.0.0.1:8080, motor at 50 %
   ```
2. **Terminal 2: run the same tests you will run on the real device**
   ```bash
   python test_shelly.py --host 127.0.0.1:8080                         # connection + status
   python test_shelly.py --host 127.0.0.1:8080 --move open             # asks 'yes', runs 1 s
   python test_shelly.py --host 127.0.0.1:8080 --move open --back --yes  # there and back
   ```
3. **Stop both with Ctrl+C.** The simulated motor halts where it is (like a power cut) and the port is freed.

Simulate harder cases:

```bash
python modules/fake_shelly.py --start-pos 98                    # a move runs into the open limit switch
python modules/fake_shelly.py --close-seconds 15                # closes faster: the timed return overshoots
python modules/fake_shelly.py --uncalibrated --password secret  # like a new device with a password
python modules/fake_shelly.py --watts 0                         # dead motor: draws no power
python modules/fake_shelly.py --port 8081                       # a second simulator next to the first
```

## Workflow 2: test the real Shelly and motor

The Jetson Orin runs the model and logic; a **Shelly 2PM Gen4** in its **cover** profile switches the
**SL45RM-50/12** tubular motor (230 V AC): output O1 runs it one way (`open`), O2 the other (`close`).
The Jetson talks to the Shelly over the network with the Shelly RPC API (HTTP).

**Before the first move**
- Find the Shelly's IP address in the Shelly app or your router, and check the Jetson reaches it: `ping <ip>`.
- The 2PM drives motors with **two direction wires** (4 wires: up, down, neutral, earth). A tubular motor with a
  built-in radio receiver only has L, N and PE and cannot be driven by the 2PM's outputs; check the motor's label or wires.
- Mains wiring (230 V) should be done or checked by a qualified electrician, following Shelly's cover wiring diagram.
- Keep clear of the umbrella while testing, and keep a second terminal ready for the [emergency stop](#stopping-safely).

**Step by step** (replace `192.168.1.50` with your Shelly's address):

1. **Read-only check.** Never moves the motor.
   ```bash
   python test_shelly.py --host 192.168.1.50
   ```
   Prints device info (model, firmware, profile, authentication), the cover config (max run times, power limit,
   obstruction detection) and the live status (state, position, power, voltage, temperature, errors).
2. **Password** (only if the check says authentication is on):
   ```bash
   python test_shelly.py --host 192.168.1.50 --check-auth --password 'secret'   # is this the right one?
   export SHELLY_PASSWORD='secret'                                             # then every command uses it
   ```
3. **Cover profile.** The Shelly must be in the `cover` profile. Set it in the Shelly app, or:
   ```bash
   python test_shelly.py --host 192.168.1.50 --set-cover-profile   # reboots it; schedules and webhooks are lost
   ```
4. **First move: 1 s one way.** Asks you to type `yes` first.
   ```bash
   python test_shelly.py --host 192.168.1.50 --move open
   ```
   Check the **peak power** it prints. Near 0 W means the motor was not driven: wiring, a limit switch already
   reached in that direction (try `--move close`), or a motor the 2PM cannot drive (see above).
   If `open` turns the wrong way, swap the O1/O2 motor wires, or reverse the direction in the Shelly app.
5. **Round trips.** Move, then return to where it started:
   ```bash
   python test_shelly.py --host 192.168.1.50 --move open --back
   python test_shelly.py --host 192.168.1.50 --move close --back --cycles 3 --duration 2
   ```

A move lasts at most 3 s (`--duration`, default 1 s, about 0.2 turn at 12 rpm). The status is printed every
0.1 s while it runs, and every move ends with `Cover.Stop`. The duration is also sent to the Shelly, so it stops
the motor by itself if the Jetson loses the connection.

### How the round trip (`--back`) returns to the start

1. Records the start: state and, if the Shelly is calibrated, the position (0 % closed to 100 % open).
2. Runs `--move` for `--duration` and measures how long the motor really turned. That is shorter when a
   limit switch stops it, so the return never overshoots the start.
3. Runs the other way for exactly that time.
4. If calibrated and the position is more than 1 % off (motors often run a little faster one way under load),
   moves to the exact start position with `Cover.GoToPosition`.
5. Prints the state and position before and after, `same` or `DIFFERENT`.

Without calibration the Shelly cannot report the position, so the return is by time only (within about 0.1 s) and
the script asks you to check by eye. Calibrate once in the Shelly app (or `Cover.Calibrate`) for exact returns;
the Shelly then runs the umbrella fully between both limit switches to learn the travel time.

### `test_shelly.py` exit codes

| Code | Meaning |
|---|---|
| 0 | OK (connection fine, move done, round trips back at the start, or stop confirmed) |
| 1 | cancelled at the `yes` prompt, or `--check-auth`: no password given / password wrong |
| 2 | failed: Shelly unreachable, wrong password, wrong profile, Shelly errors, or `--stop` not confirmed |
| 3 | a round trip did not end where it started |
| 130 | stopped with Ctrl+C (motor stopped first) |

### Password

The Shelly never reveals its password; `--check-auth` tells you whether one is set and tests the one you give.
If it is lost: set a new one in the Shelly app (if the device is linked to your account), or factory-reset
the Shelly with its button as described in the Shelly manual and set it up again.
If none is set, anyone on the network can move the umbrella: set one in the Shelly app or web UI (Settings > Authentication).

## Workflow 3: replay held-out weather through the model

`run_simulation.py` loads the model and the clips in `evaluation_dataset/` and feeds them to the controller one
20 s reading at a time, as the device would receive them:

```bash
python run_simulation.py                          # every clip, every reading
python run_simulation.py --clip 20240102_0730     # one clip (repeatable)
python run_simulation.py --category heat --limit 3
python run_simulation.py --every 3 --features     # one line per minute, plus window features
python run_simulation.py --speed 60               # paced at 60x real time
python run_simulation.py --moves-only --out sim.csv
```

```text
time                 phase   |   temp   hum     wind rain   UTCI | P(OPEN) state | decision | label  in 5m
2024-01-02 07:36:00  lead_in |   3.4C   96%  0.0 m/s WET    5.2C |   0.728     7 | OPENS    | OPEN   OPEN     (warming up)
2024-01-02 08:00:00  eval    |   3.8C   97%  0.3 m/s dry    5.6C |   0.542     7 | OPEN     | CLOSED CLOSED  MISS
2024-01-02 08:00:20  eval    |   3.8C   97%  0.4 m/s dry    5.6C |   0.023     3 | CLOSES   | CLOSED CLOSED  ok
  umbrella moved 2x (label 2x), open 24 min (label 21 min) | scored rows: 180, agree 99.4%
```

- **Inputs:** temperature, humidity, wind, rain detector, UTCI; `--features` adds the 10-min mean UTCI, 5-min wet share, rain recency and 15-min humidity change.
- **Inference:** P(OPEN in 5 min), the most likely weather state, and the decision (`OPENS` / `CLOSES` mark a move).
- **Check:** the label now and 5 min ahead; on scored (`eval`) rows, `ok` or `MISS` against the label ahead.
- With `--every N`, moves, restarts and misses are still always printed.
- Each clip ends with a summary (moves, minutes open, agreement); the run ends with scores per category against
  persistence ("the label stays as it is now").
- Colours are used in a terminal; `--no-color` turns them off.

For scores only, without the per-reading output:

```bash
python -m modules.inference                         # all clips: scores vs persistence, per category
python -m modules.inference --clip 20240102_0730    # one clip: every open/close move
python -m modules.inference --out predictions.csv   # also save the decision for every row
```

Scores are on the `eval` rows against `label_ahead` (the label 5 min later); `false_open` is the share of
should-be-CLOSED rows where the umbrella was open.

### Current results on the 60 default clips

| | F1 | recall | false_open |
|---|---|---|---|
| model | 0.60 | 0.47 | 0.07 |
| persistence | 0.89 | 0.89 | 0.07 |

By category the model is close to persistence on rain (F1 0.75 vs 0.78) but misses heat (recall 0.11).
The saved model was trained on summer data only (2026-05-09 to 2026-07-07) and its OPEN threshold is 0.7,
so it does worse here than the walk-forward results in `training.ipynb`, which retrain weekly.

## Workflow 4: refresh the data or the model

After re-running the notebooks:

```bash
python extract_evaluation.py                                        # new clips -> evaluation_dataset/
python -m modules.inference --export ../models/umbrella_hmm.joblib  # new model -> models/umbrella_hmm.npz
python run_simulation.py --every 30                                 # check the result
```

`extract_evaluation.py` rebuilds the train/evaluation split of `training.ipynb`, so **no extracted row was ever
used for training**. Its split options (`--n-hours 3 --split-seed 42 --gap-minutes 30 --earliest-hour 6`, the
defaults) must match the notebook's. Other selections:

```bash
python extract_evaluation.py --all                                     # every held-out clip (~1,560)
python extract_evaluation.py --rain 40 --heat 30 --mixed 5 --dry 10 --seed 1
python extract_evaluation.py --out /tmp/clips                          # elsewhere; keeps evaluation_dataset/ as is
```

Each clip is one unbroken, real stretch of weather at 20 s steps, never cut mid-event:

| Phase | Length | Use |
|---|---|---|
| `lead_in` | 30 min | warm-up: fills the 15-min feature windows, as on a device that was already running |
| `eval` | 60 min (or more when evaluation hours touch) | the rows to score |
| `gap` | between two evaluation hours, if any | unscored |
| `tail` | 30 min | lets the umbrella finish closing; holds the labels for the last scored rows |

Clips are grouped by why the umbrella is OPEN: `rain`, `heat`, `heat+rain` (`--mixed`), or `dry` (stays CLOSED).
Within a category it picks clips where the umbrella both **opens and closes** first,
then clips with only one of the two, and spreads them across months.
Heat clips only exist in summer and `heat+rain` clips are rare (7 in total), so those categories cover fewer months.

`evaluation_dataset/clips.csv`, one row per 20 s step, sorted by time:

| Column | Meaning |
|---|---|
| `timestamp`, `clip_id`, `phase` | time, clip (its start, `YYYYMMDD_HHMM`), phase above |
| `temperature`, `humidity`, `wind_speed`, `wind_dir`, `rain_flag` | device sensors (rain detector 0 = dry, 10 = wet) |
| `utci` | UTCI computed from the sensors (as the device would) |
| `label`, `reason`, `label_p_open` | ground truth now: OPEN/CLOSED, why, and the label's probability |
| `label_ahead` | the label 5 min later, what the model predicts; empty in the last 5 min of a clip |

`evaluation_dataset/manifest.csv`, one row per clip: start, end, minutes per phase, category,
share OPEN, number of openings and closings, `full_cycle` (opens and closes), max UTCI, share wet.

---

## Stopping safely

**Ctrl+C is always safe**, in every script. Closing the terminal, `kill <pid>` and Ctrl+Break do the same
in `test_shelly.py` and the simulator.

| Script | What happens on Ctrl+C |
|---|---|
| `test_shelly.py` | the motor is stopped **before** the script exits, then the final state is printed (exit code 130) |
| `modules/fake_shelly.py` | the simulated motor halts where it is (power cut), its position is printed, the port is freed |
| `run_simulation.py` | stops at once; the clip in progress is dropped, finished clips are scored and saved to `--out` |
| `extract_evaluation.py` | nothing is written; the existing `clips.csv` / `manifest.csv` stay as they were |
| `modules/inference.py` | nothing is written; `--out` and the exported model are never left half-written |

**With the real motor (`test_shelly.py`):**

1. `Cover.Stop` is sent right away, retried up to 3 times (1.5 s each) if the Shelly does not answer.
   Pressing Ctrl+C again while it is being sent does **not** cancel it ("stopping the motor first, please wait").
2. The final state is printed (`state`, `pos`, `power`). An interrupted round trip may not be back at its start:
   move it back with `--move open|close` or the Shelly app.
3. If the Shelly cannot be reached, the script says so. The Shelly still stops by itself after the `--duration`
   it was given (every move command carries it), or at its maximum run time. If the motor keeps running, cut its power.

A stop is also sent if the script fails for any other reason after a move started.

**Emergency stop from another terminal** (or when no script is running):

```bash
python test_shelly.py --host 192.168.1.50 --stop
```

**Files are never left half-written:** output files (`--out`, the extracted CSVs, the exported model) are written
in full to a temporary file next to the target and only then swapped in.

**Simulator ports:** a second simulator on a port that is already in use is refused with a clear message
(start it with `--port 8081`), so commands never reach the wrong one.

---

## Options reference

`python <script> --help` prints the same list.

### `test_shelly.py`

| Option | Default | Meaning |
|---|---|---|
| `--host HOST` | **required** | Shelly IP address or `hostname[:port]`, e.g. `192.168.1.50` or `127.0.0.1:8080` |
| `--password` | `$SHELLY_PASSWORD`, else none | password if authentication is on (user is always `admin`) |
| `--timeout` | 3 | seconds per request |
| `--check-auth` | off | only check whether a password is set and whether `--password` is right |
| `--move open\|close` | none (read-only) | run the motor this way |
| `--duration` | 1 | seconds to run, 0.1 to 3 |
| `--back` | off | round trip: after the move, return to the start state |
| `--cycles` | 1 | round trips to make with `--back`, 1 to 10 |
| `--stop` | off | emergency stop: only send `Cover.Stop` |
| `--yes` | off (asks) | do not ask for confirmation before moving |
| `--set-cover-profile` | off | switch the Shelly to the cover profile if needed; reboots it |

### `modules/fake_shelly.py`

| Option | Default | Meaning |
|---|---|---|
| `--port` | 8080 | port on 127.0.0.1 |
| `--password` | none | require this password (user `admin`) |
| `--start-pos` | 50 | start position, 0–100 % |
| `--open-seconds` | 20 | full travel time opening |
| `--close-seconds` | 20 | full travel time closing |
| `--watts` | 120 | power while turning; 0 = dead motor |
| `--uncalibrated` | off (calibrated) | do not report the position, like a new device |

### `run_simulation.py`

| Option | Default | Meaning |
|---|---|---|
| `--model` | `models/umbrella_hmm.npz` | model file |
| `--clips` | `evaluation_dataset/clips.csv` | clips to replay (`manifest.csv` is read from the same folder) |
| `--clip ID` | all clips | only this clip, repeatable |
| `--category` | all | only `rain`, `heat`, `heat+rain` or `dry` clips |
| `--limit N` | no limit | at most N clips |
| `--every N` | 1 | print every Nth reading (moves, restarts and misses always) |
| `--moves-only` | off | print only the readings where the umbrella moves |
| `--features` | off | also print the main window features |
| `--speed X` | 0 (as fast as possible) | pace at X times real time |
| `--out FILE` | not saved | save every decision to this CSV |
| `--no-color` | off | no colours |

### `extract_evaluation.py`

| Option | Default | Meaning |
|---|---|---|
| `--labeled` | `../labeled_dataset/` | labeled 20 s data from `labeling.ipynb` |
| `--out` | `evaluation_dataset/` | folder for `clips.csv` and `manifest.csv` |
| `--rain` / `--heat` / `--mixed` / `--dry` | 20 / 20 / 5 / 15 | clips per category |
| `--all` | off | every held-out clip instead of the quotas |
| `--seed` | 0 | seed for picking clips |
| `--n-hours` | 3 | evaluation hours per day (must match `training.ipynb`) |
| `--split-seed` | 42 | split seed (must match) |
| `--gap-minutes` | 30 | buffer around each evaluation hour (must match) |
| `--earliest-hour` | 6 | no evaluation hours before this hour (must match) |
| `--horizon-minutes` | 5 | how far ahead `label_ahead` looks (the model's `HORIZON_MINUTES`) |

### `python -m modules.inference`

| Option | Default | Meaning |
|---|---|---|
| `--model` | `models/umbrella_hmm.npz` | `.npz` (or `.joblib`) model |
| `--clips` | `evaluation_dataset/clips.csv` | clips to replay |
| `--clip ID` | all clips | only this clip, repeatable |
| `--out FILE` | not saved | write the per-row decisions to this CSV |
| `--export JOBLIB` | off | convert a training `.joblib` bundle to `--model` (`.npz`) and exit |

---

## Runtime: `modules/`

Self-contained: the device needs only `deployment/` (numpy, pandas, and pythermalcomfort for UTCI), not the training code.

```text
deployment/
├── run_simulation.py, test_shelly.py, extract_evaluation.py
├── modules/
│   ├── dataset.py      loading and formatting: Reading, load_clips, load_manifest, iter_clips, compute_utci, save_csv
│   ├── inference.py    UmbrellaModel, WindowFeatures, Umbrella (controller), run_stream, summarize, CLI
│   └── fake_shelly.py  simulated Shelly 2PM Gen4 + motor, to run test_shelly.py without hardware
├── models/umbrella_hmm.npz      the trained model as plain arrays
└── evaluation_dataset/          held-out clips (Workflow 4)
```

Every 20 s the controller does what the training pipeline does, one reading at a time:

```text
Reading -> UTCI -> window features (last 15 min) -> HMM forward step
        -> P(OPEN in 5 min) -> open at >= 0.7, close below 0.5, hold 3 min -> OPEN / CLOSED
```

It gives the same features, probabilities and decisions as `training.ipynb`
(checked on all 23,193 clip rows: differences below 1e-13, no decision differs).
A time jump (anything but one 20 s step) restarts it, like a data gap in training; the first 15 min after a restart are flagged `warming_up`.

### Live use (Python)

```python
from modules.dataset import Reading
from modules.inference import Umbrella, UmbrellaModel

umbrella = Umbrella(UmbrellaModel.load())       # models/umbrella_hmm.npz

# every 20 s, from the sensors (UTCI is computed when not given):
decision = umbrella.step(Reading.from_dict({
    "timestamp": "2026-07-07 14:00:20", "temperature": 31.2, "humidity": 62.0,
    "wind_speed": 1.4, "rain_flag": 0,
}))
decision.open, decision.p_open      # True/False, P(OPEN in 5 min)
```

Replaying recorded data in Python:

```python
from modules.dataset import load_clips, load_manifest
from modules.inference import UmbrellaModel, run_stream, summarize

clips = load_clips()                             # evaluation_dataset/clips.csv
predictions = run_stream(UmbrellaModel.load(), clips)
summarize(predictions, load_manifest())
```
