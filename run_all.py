"""
run_all.py
Run the complete TrafficADS project pipeline end-to-end.
Usage:  python run_all.py
"""

import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

steps = [
    ("Generating Datasets",         "generate_dataset.py"),
    ("Training Traffic Model",       "traffic_model.py"),
    ("Pothole Detection Module",     "pothole_detection.py"),
    ("Landmark Detection Module",    "landmark_detection.py"),
    ("Route Optimization Module",    "route_optimizer.py"),
]

print("\n" + "=" * 60)
print("   TrafficADS — Full Pipeline Runner")
print("=" * 60)

for step_name, script in steps:
    print(f"\n{'─'*60}")
    print(f"▶  {step_name}")
    print(f"{'─'*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Failed at: {script}")
        sys.exit(1)

print("\n" + "=" * 60)
print("  ✅ All modules complete!")
print("  Outputs saved to: outputs/")
print("  Models saved to : models/")
print("\n  To launch the web dashboard run:")
print("    python dashboard.py")
print("  Then open: http://localhost:5000")
print("=" * 60 + "\n")
