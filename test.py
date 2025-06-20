import os
from datetime import datetime
import torch
from model import get_model
from data import prepare_test_data, create_data_loader, SEED, LOCAL_BERT_DIR, MAX_SEQ_LEN, BATCH_SIZE, standard_disease_to_idx
from transformers import BertTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score



# 模型名字
model_name = 'bert_classifier'

# 设置保存目录
save_dir = 'model_save'
log_save_dir = 'log_save'

# 获取当前时间
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

# 日志和模型文件名称
log_save_name = LOCAL_BERT_DIR + f'_seed={SEED}_{time_str}.txt'
model_save_name = 'augmented_bert-base-chinese_seed=40_2024-09-12_18-04-39_best_model_state.pt'

    

# 日志写入函数
def frwirt(context):
    with open(os.path.join(log_save_dir, log_save_name), 'a') as f:
        f.write(context + '\n')

# 将时间和文件名写入日志
frwirt(time_str)
frwirt(model_save_name)
print(model_save_name)

# 类别数量
n_classes = len(standard_disease_to_idx)
print(f'类别数量: {n_classes}')

# 加载模型函数
def load_model(model_path):
    model = get_model(model_name, n_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# 测试函数
def test(model, data_loader, device):
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, macro_f1, macro_precision, macro_recall

# 主函数
def main(model_path, test_data_path):
    # 加载测试数据
    test_texts, test_labels = prepare_test_data(test_data_path)

    tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_DIR)

    test_data_loader = create_data_loader(test_texts, test_labels, tokenizer, MAX_SEQ_LEN, BATCH_SIZE, shuffle=False)

    model = load_model(model_path)

    test_acc, macro_f1, macro_pre, macro_rec = test(model, test_data_loader, device)

    frwirt(f'Test accuracy: {test_acc.item()} macro f1: {macro_f1} macro precision: {macro_pre} macro recall: {macro_rec} 测试完成! ')
    print(f'Test accuracy: {test_acc.item()} macro f1: {macro_f1} macro precision: {macro_pre} macro recall: {macro_rec} 测试完成!')

if __name__ == '__main__':
    # 设置设备，匹配训练脚本中的cuda:1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # 加载模型路径
    model_path = os.path.join(save_dir, model_save_name)   
    test_data_path = 'data/test.csv'  # 测试数据文件路径
    main(model_path, test_data_path)
