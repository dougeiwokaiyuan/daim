import subprocess

# 定义输入和输出文件名
input_file = "HTTP.ui"
output_file = "THHP_ui.py"

# 构建命令，这里包含了 -x 参数，如果不需要测试代码块，可以去掉 -x 及其后面内容
command = f"pyuic6 -x {input_file} -o {output_file}"

try:
    # 执行命令
    subprocess.run(command, shell=True, check=True)
    print(f"Successfully converted {input_file} to {output_file}")
except subprocess.CalledProcessError as e:
    print(f"Error converting file: {e}")
