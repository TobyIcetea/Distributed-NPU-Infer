import json
import datetime
import re
import argparse
import subprocess
import os
import sys
import time
from kubernetes import client, config, utils
from kubernetes.client import ApiException


def check_jobs_complete(namespace='default'):
    """
    检查指定命名空间中的所有Jobs是否都已完成

    :param namespace: 要检查的命名空间，如果为None则检查所有命名空间
    :return: 如果所有Jobs都完成返回True，否则返回False
    """
    # 加载kube配置
    try:
        config.load_kube_config()  # 用于本地开发环境
    except:
        config.load_incluster_config()  # 用于集群内部运行

    batch_v1 = client.BatchV1Api()

    # 获取Jobs列表
    if namespace:
        jobs = batch_v1.list_namespaced_job(namespace)
    else:
        jobs = batch_v1.list_job_for_all_namespaces()

    all_complete = True

    for job in jobs.items:
        job_name = job.metadata.name
        namespace = job.metadata.namespace

        # 检查Job的状态
        if job.status.succeeded is None or job.status.succeeded < job.spec.completions:
            print(f"Job {namespace}/{job_name} is not complete (succeeded: {job.status.succeeded or 0}/{job.spec.completions})")
            all_complete = False
        else:
            print(f"Job {namespace}/{job_name} is complete")

    return all_complete

# 等价于 kubectl delete -f ${filename}
def delete_k8s_resource(filename):
    """
    使用kubectl delete -f命令删除指定的Kubernetes资源文件。
    """
    # 构建命令参数列表（避免shell注入）
    command = ['kubectl', 'delete', '-f', filename]

    # 执行命令并检查结果
    subprocess.run(command, check=True)


def monitor_jobs_and_delete_file(namespace, filename, interval):
    """
    持续监测Jobs直到完成，然后删除指定文件

    :param namespace: 要监测的命名空间
    :param filename: 要删除的文件名
    :param interval: 检查间隔时间(秒)
    """
    print(f"开始监测Jobs，完成后将删除文件: {filename}")

    while True:
        try:
            if check_jobs_complete(namespace):
                print("所有Jobs已完成，准备删除文件...")
                delete_k8s_resource(filename)
                try:
                    os.remove(filename)
                    print(f"文件 {filename} 已成功删除")
                except FileNotFoundError:
                    print(f"文件 {filename} 不存在，无需删除")
                except Exception as e:
                    print(f"删除文件时出错: {str(e)}")
                break
            else:
                print(f"Jobs未全部完成，等待...")
                time.sleep(interval)
        except Exception as e:
            print(f"监测Jobs时出错: {str(e)}")
            print(f"等待{interval}秒后重试...")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description='Generate Kubernetes jobs with unique names')
    parser.add_argument('--json', type=str, required=True, help='NPU levels JSON config path')
    parser.add_argument('--yaml', type=str, required=True, help='Original YAML template path')
    args = parser.parse_args()

    # 文件验证
    for path in [args.json, args.yaml]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # 读取配置
    with open(args.json, 'r') as f:
        nodes = json.load(f).keys()

    with open(args.yaml, 'r') as f:
        yaml_template = f.read()

    # 生成带编号的配置
    job_sections = []
    for i, node in enumerate(nodes, 1):
        # 同时替换nodeName和metadata.name
        modified = yaml_template
        modified = re.sub(
            r'^(\s*name:\s*)(\S+)$',
            rf'\1\2-{i}',  # 在原有名称后添加编号
            modified,
            flags=re.MULTILINE,
            count=1  # 只替换第一个匹配项（metadata部分）
        )
        modified = re.sub(
            r'nodeName:\s*\S+',
            f'nodeName: {node}',
            modified
        )
        job_sections.append(modified)

    # 生成文件名并写入
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"ascend-infer-job-{timestamp}.yaml"
    with open(output_file, 'w') as f:
        f.write('\n---\n'.join(job_sections))

    # 执行kubectl
    kubectl_cmd = ['kubectl', 'apply', '-f', output_file]
    result = subprocess.run(kubectl_cmd, check=True, capture_output=True, text=True)

    print(f"Successfully applied jobs:\n{result.stdout}")

    # 检查 jobs 是否完成
    namespace = 'default'
    monitor_jobs_and_delete_file(namespace, output_file, 5)



if __name__ == "__main__":
    main()
