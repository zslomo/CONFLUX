import sys
import os
import time
from loguru import logger
from configparser import ConfigParser
from src.sample_processing import SampleProcess
from src.student_model import StudentModel
from src.teacher_model import TeacherModel
from src.utils import Utils


def run(cfg):
    '''
    No config file demo
    '''
    date = cfg.get('run', 'date')
    if date == "None":
        date = time.strftime('%Y%m%d', time.localtime())
    
    train_type = cfg.get('run', 'train_type')
    date_pre = cfg.get('run', 'date_pre')

    traffic_type = cfg.get('train_param', 'traffic_type')
    cross_num = int(cfg.get('train_param', 'cross_num'))
    epoches = int(cfg.get('train_param', 'epoches'))
    batch_size = int(cfg.get('train_param', 'batch_size'))
    lr = float(cfg.get('train_param', 'lr'))
    early_stop = int(cfg.get('train_param', 'early_stop'))
    net_params = cfg.get('train_param', 'net_params')
    process_cnt = cfg.get('train_param', 'process_cnt')
    load_model = cfg.get('train_param', 'load_model') == 'True'
    save_model = cfg.get('train_param', 'save_model') == 'True'
    train_mode = cfg.get('train_param', 'train_mode')
    vsibile_epoch = int(cfg.get('train_param', 'vsibile_epoch'))
    model_type = cfg.get('train_param', 'model_type')

    res_path = cfg.get('data_path', 'res_path')
    generate_train_sample_file = cfg.get('data_path', 'generate_train_sample_file')
    generate_valid_sample_file = cfg.get('data_path', 'generate_valid_sample_file')
    train_sample_name = cfg.get('data_path', 'train_sample_name').format(traffic_type)
    test_sample_name = cfg.get('data_path', 'test_sample_name').format(traffic_type)
    save_dir_name = cfg.get('data_path', 'save_dir_name')

    feature_mapping_file = cfg.get('data_path', 'feature_mapping_file')
    targeting_feature_file = cfg.get('data_path', 'targeting_feature_file')

    sample_rate = int(cfg.get('exp_info', 'sample_rate'))

    logger.info("---------------------------run-------------------------------")
    logger.info("date_pre: {}".format(date_pre))
    logger.info("save_model: {}".format(save_model))
    logger.info("-----------------------train params--------------------------")
    logger.info("cross_num: {}".format(cross_num))
    logger.info("train_type: {}".format(train_type))
    logger.info("train_type: {}".format(train_type))
    logger.info("epoches: {}".format(epoches))
    logger.info("lr: {}".format(lr))
    logger.info("early_stop: {}".format(early_stop))
    logger.info("net_params: {}".format(net_params))
    logger.info("process_cnt: {}".format(process_cnt))
    logger.info("load_model: {}".format(load_model))
    logger.info("train_mode: {}".format(train_mode))
    logger.info("vsibile_epoch: {}".format(vsibile_epoch))
    logger.info("model_type: {}".format(model_type))
    logger.info("sample_rate: {}".format(sample_rate))
    logger.info("------------------------data path-----------------------------")
    logger.info("res_path: {}".format(res_path))
    logger.info("generate_train_sample_file: {}".format(generate_train_sample_file))
    logger.info("generate_valid_sample_file: {}".format(generate_valid_sample_file))
    logger.info("train_sample_name: {}".format(train_sample_name))
    logger.info("test_sample_name: {}".format(test_sample_name))
    logger.info("save_dir_name: {}".format(save_dir_name))
    logger.info("------------------------exp info-----------------------------")

    logger.info("feature_mapping_file: {}".format(feature_mapping_file))
    logger.info("targeting_feature_file: {}".format(targeting_feature_file))
    sp_class = SampleProcess(logger)
    teacher_model_class = TeacherModel(logger)
    student_model_class = StudentModel(logger)
    utils_class = Utils(logger)

    logger.info("train_sample_name = {}, test_sample_name = {}".format(train_sample_name, test_sample_name))

    targeting_feature_file = targeting_feature_file.format(date)
    train_samples = utils_class.get_pre_sample(train_sample_name)
    test_samples = utils_class.get_pre_sample(test_sample_name)
    # feature映射文件
    feature_mapping = utils_class.get_json_file(feature_mapping_file)
    # 广告定向
    targeting_feature_dict = utils_class.get_json_file(targeting_feature_file)

    params = targeting_feature_dict, feature_mapping, generate_train_sample_file, batch_size
    sp_class.sample_multiple_process(train_samples, params, process_cnt=process_cnt)
    sp_class.sample_multiple_process(test_samples, params, process_cnt=process_cnt)

    logger.info("multi process done.")
    train_file_names = [
        os.path.join(generate_train_sample_file, f) for f in os.listdir(generate_train_sample_file)
        if 'swp' not in f
    ]
    valid_file_names = [
        os.path.join(generate_valid_sample_file, f) for f in os.listdir(generate_valid_sample_file)
        if 'swp' not in f
    ]
    train_samples = utils_class.load_pickle_samples(train_file_names)
    valid_samples = utils_class.load_pickle_samples(valid_file_names)

    samples = [train_samples, valid_samples]

    # teacher_model
    teacher_res_path = os.path.join(res_path, "teacher_output")
    teacher_save_dir_name = os.path.join(save_dir_name, "teacher")
    teacher_model_op = teacher_model_class.build_network(lr, net_params, cross_num, with_cross=True)
    teacher_model_class.train(samples, teacher_model_op, epoches, early_stop, save_model, teacher_save_dir_name)
    teacher_model_class.inference(save_dir_name, samples, teacher_res_path)
    
    # student_model
    student_save_dir_name = os.path.join(save_dir_name, "student")
    student_model_op = student_model_class.build_network(lr, net_params)
    student_model_class.train(samples, student_model_op, epoches, early_stop, teacher_res_path, save_model, student_save_dir_name)


if __name__ == '__main__':
    config_file = sys.argv[1]
    cfg = ConfigParser()
    cfg.read(config_file, encoding='iso-8859-1')
    run_name = cfg.get('run', 'run_name')
    logger.add("./log/train/" + run_name + "/_{time}.log", rotation="5000 MB", level="INFO")
    run(cfg)
