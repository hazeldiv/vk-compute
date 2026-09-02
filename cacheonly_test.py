import subprocess
p = subprocess.run([r"D:\coding\python\vk_engine\bin\main.exe", "--weights", r"D:\coding\python\vk_engine\model\Qwen3.5-9B-weight", "--max-ctx", "8192", "--max-new", "32"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, cwd=r"D:\coding\python\vk_engine\bin")
print("RC:", p.returncode)
print("STDERR:", p.stderr[-600:])
