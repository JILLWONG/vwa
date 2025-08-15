import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import re

def model_mapper(model):
    if model == 'qwen25vl72b':
        return 'Qwen2.5VL-72B'
    elif model == 'qwen25vl7b':
        return 'Qwen2.5VL-7B'
    else:
        raise ValueError('Bad model name')

def method_mapper(method):
    if method == 'image':
        return 'Multimodel Image + Caps + Acc. Tree'
    elif method == 'image_som':
        return 'Multimodel (SoM) Image + Caps + SoM'
    else:
        raise ValueError('Bad method name')

def get_domain_results(domain_folder):
    result_file = f'{domain_folder}/result.json'
    if u.is_file_exist(result_file): 
        domain_results = u.read_json(result_file)
        return domain_results
    files = u.list_files(domain_folder, True)
    pattern = re.compile(r"^result_\d+_\d+\.json$")
    result_files = [a for a in files if pattern.match(u.get_name(a))]
    if result_files:
        domain_results = {}
        for result_file in result_files:
            result_info = u.read_json(result_file)
            domain_results.update(result_info)
        return domain_results
    result_folder = f'{domain_folder}/results/'
    if u.is_folder_exist(result_folder):
        domain_results = {}
        result_files = u.list_files(domain_folder, True)
        result_files = [a for a in result_files if '.json' in a]
        for result_file in result_files:
            result = u.read_json(result_file)
            domain_results.update(result)
        return domain_results

if __name__ == "__main__":
    vwa_data_path = f'{u.get_nas()}/gui_dataset/visualwebarena/'
    result_path = f'{vwa_data_path}/saved_results/'
    folders = u.list_folder_names(result_path)
    folders = [a for a in folders if 'auth' not in a]
    domains = {'reddit': 210, 'classifieds': 234, 'shopping': 466}
    metrics = {}
    for job_name in folders:
        job_folder = f'{result_path}/{job_name}/'
        if u.is_folder_empty(job_folder): continue
        metrics[job_name] = {}
        pass_count = 0
        for domain in domains.keys():
            domain_folder = f'{job_folder}/{domain}'
            if not u.is_folder_exist(domain_folder): continue
            domain_results = get_domain_results(domain_folder)
            if len(domain_results) != domains[domain]: continue
            domain_pass_count = sum(1 for v in domain_results.values() if v == 1)
            domain_avg = domain_pass_count / len(domain_results)
            metrics[job_name][domain] = domain_avg
            pass_count += domain_pass_count
        avg = pass_count / sum(domains.values())
        metrics[job_name]['avg'] = avg

    for job_name in list(metrics.keys()):
        if metrics[job_name]['avg'] == 0.0: del metrics[job_name]
    
    u.pl()
    for job_name in list(metrics.keys()):
        values = metrics[job_name]
        idx = job_name.find('_')
        model = model_mapper(job_name[:idx])
        method = method_mapper(job_name[idx+1:])
        classifieds = values['classifieds'] * 100
        reddit = values['reddit'] * 100
        shopping = values['shopping'] * 100
        avg = values['avg'] * 100
        latex_string = f'ours & {method} & {model} & {classifieds:.2f} & {reddit:.2f} & {shopping:.2f} & {avg:.2f} \\\\'
        print(latex_string)

    # u.print_json(metrics)
