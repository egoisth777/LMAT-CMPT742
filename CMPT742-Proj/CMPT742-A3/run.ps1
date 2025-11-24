param(
    [ValidateSet("train", "test", "map", "gen_report", "setup")]
    [string]$Mode = "train"
)

$python = "python"

switch ($Mode) {
    "train" {
        & $python "main.py"
    }
    "test" {
        & $python "main.py" "--test"
    }
    "map" {
        & $python "gen_map.py"
    }
    "gen_report" {
        & $python "visualize.py"
    }
    "setup" {
        if (-not (Test-Path "requirements.txt")) {
            throw "requirements.txt not found in the current directory."
        }
        & $python "-m" "pip" "install" "-r" "requirements.txt"
    }
}
