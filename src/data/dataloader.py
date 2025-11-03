import torch
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import numpy as np
from tqdm import tqdm


class LocalIWSLTXMLDataLoader:
    def __init__(self, batch_size=32, max_seq_len=128, data_dir="D:/huggingface_cache/datasets--iwslt2017/en-de"):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.pad_token_id = 0
        self.sos_token_id = 3
        self.eos_token_id = 4

        # 构建tokenizer
        self.src_tokenizer, self.tgt_tokenizer = self._build_tokenizers()
        self.src_vocab_size = self.src_tokenizer.get_vocab_size()
        self.tgt_vocab_size = self.tgt_tokenizer.get_vocab_size()

        print(f"Source vocab size: {self.src_vocab_size}")
        print(f"Target vocab size: {self.tgt_vocab_size}")

    def _parse_xml_file(self, xml_file):
        """解析XML文件，提取文本"""
        sentences = []
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # IWSLT XML文件结构通常包含<seg>标签
            for seg in root.iter('seg'):
                text = seg.text.strip() if seg.text else ""
                if text:
                    sentences.append(text)

        except Exception as e:
            print(f"解析XML文件 {xml_file} 时出错: {e}")

        return sentences

    def _read_text_file(self, text_file):
        """读取纯文本文件"""
        sentences = []
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和XML标签行
                    if line and not line.startswith('<'):
                        sentences.append(line)
        except Exception as e:
            print(f"读取文本文件 {text_file} 时出错: {e}")

        return sentences

    def _get_data_splits(self):
        """获取训练集和验证集数据"""
        # 训练数据
        train_en = self._read_text_file(os.path.join(self.data_dir, "train.tags.en-de.en"))
        train_de = self._read_text_file(os.path.join(self.data_dir, "train.tags.en-de.de"))

        # 验证数据 - 使用dev2010
        val_en = self._parse_xml_file(os.path.join(self.data_dir, "IWSLT17.TED.dev2010.en-de.en.xml"))
        val_de = self._parse_xml_file(os.path.join(self.data_dir, "IWSLT17.TED.dev2010.en-de.de.xml"))

        print(f"训练集: {len(train_en)} 个英语句子, {len(train_de)} 个德语句子")
        print(f"验证集: {len(val_en)} 个英语句子, {len(val_de)} 个德语句子")

        # 确保数据对齐
        min_train = min(len(train_en), len(train_de))
        min_val = min(len(val_en), len(val_de))

        train_data = list(zip(train_en[:min_train], train_de[:min_train]))
        val_data = list(zip(val_en[:min_val], val_de[:min_val]))

        return train_data, val_data

    def _build_tokenizers(self):
        """构建英德tokenizer"""
        train_data, _ = self._get_data_splits()
        train_en, train_de = zip(*train_data) if train_data else ([], [])

        print(f"使用 {len(train_en)} 个句子对构建tokenizer")

        # 英文tokenizer
        src_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        src_tokenizer.pre_tokenizer = Whitespace()
        src_trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=1
        )

        # 德文tokenizer
        tgt_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tgt_tokenizer.pre_tokenizer = Whitespace()
        tgt_trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=1
        )

        # 训练tokenizer
        src_tokenizer.train_from_iterator(train_en, trainer=src_trainer)
        tgt_tokenizer.train_from_iterator(train_de, trainer=tgt_trainer)

        return src_tokenizer, tgt_tokenizer

    def _tokenize_text(self, text, tokenizer):
        """将文本转换为token IDs"""
        if not text.strip():
            return [self.pad_token_id] * self.max_seq_len

        try:
            # 添加特殊token
            encoding = tokenizer.encode("[SOS] " + text + " [EOS]")
            token_ids = encoding.ids
        except Exception as e:
            print(f"Tokenization错误: {e}")
            return [self.pad_token_id] * self.max_seq_len

        # 截断或填充
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            # 确保以EOS结尾
            if token_ids[-1] != self.eos_token_id:
                token_ids[-1] = self.eos_token_id
        elif len(token_ids) < self.max_seq_len:
            token_ids = token_ids + [self.pad_token_id] * (self.max_seq_len - len(token_ids))

        return token_ids

    def get_dataloaders(self):
        """获取数据加载器"""
        train_data, val_data = self._get_data_splits()

        # 处理训练数据
        processed_train = []
        for en_text, de_text in tqdm(train_data, desc="处理训练数据"):
            src_ids = self._tokenize_text(en_text, self.src_tokenizer)
            tgt_ids = self._tokenize_text(de_text, self.tgt_tokenizer)

            # 过滤掉太短的序列
            src_non_special = [x for x in src_ids if x not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]]
            tgt_non_special = [x for x in tgt_ids if x not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]]

            if len(src_non_special) >= 2 and len(tgt_non_special) >= 2:
                processed_train.append({
                    "src_ids": torch.tensor(src_ids),
                    "tgt_ids": torch.tensor(tgt_ids)
                })

        # 处理验证数据
        processed_val = []
        for en_text, de_text in tqdm(val_data, desc="处理验证数据"):
            src_ids = self._tokenize_text(en_text, self.src_tokenizer)
            tgt_ids = self._tokenize_text(de_text, self.tgt_tokenizer)

            src_non_special = [x for x in src_ids if x not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]]
            tgt_non_special = [x for x in tgt_ids if x not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]]

            if len(src_non_special) >= 2 and len(tgt_non_special) >= 2:
                processed_val.append({
                    "src_ids": torch.tensor(src_ids),
                    "tgt_ids": torch.tensor(tgt_ids)
                })

        print(f"处理后训练样本: {len(processed_train)}")
        print(f"处理后验证样本: {len(processed_val)}")

        # 打印一些样本用于调试
        if len(processed_train) > 0:
            sample_src = self.src_tokenizer.decode(processed_train[0]["src_ids"].tolist(), skip_special_tokens=True)
            sample_tgt = self.tgt_tokenizer.decode(processed_train[0]["tgt_ids"].tolist(), skip_special_tokens=True)
            print(f"\n训练样本示例:")
            print(f"  英语: {sample_src}")
            print(f"  德语: {sample_tgt}")

        # 创建数据加载器
        train_loader = DataLoader(processed_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(processed_val, batch_size=self.batch_size)

        return train_loader, val_loader

    def get_test_dataloaders(self, test_years=['2010', '2011', '2012', '2013', '2014', '2015']):
        """获取测试集数据加载器"""
        test_loaders = {}

        for year in test_years:
            try:
                # 尝试加载指定年份的测试集
                test_en = self._parse_xml_file(os.path.join(self.data_dir, f"IWSLT17.TED.tst{year}.en-de.en.xml"))
                test_de = self._parse_xml_file(os.path.join(self.data_dir, f"IWSLT17.TED.tst{year}.en-de.de.xml"))

                print(f"测试集 {year}: {len(test_en)} 个英语句子, {len(test_de)} 个德语句子")

                # 确保数据对齐
                min_test = min(len(test_en), len(test_de))
                test_data = list(zip(test_en[:min_test], test_de[:min_test]))

                # 处理测试数据
                processed_test = []
                for en_text, de_text in tqdm(test_data, desc=f"处理测试集 {year}"):
                    src_ids = self._tokenize_text(en_text, self.src_tokenizer)
                    tgt_ids = self._tokenize_text(de_text, self.tgt_tokenizer)

                    # 过滤掉太短的序列
                    src_non_special = [x for x in src_ids if
                                       x not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]]
                    tgt_non_special = [x for x in tgt_ids if
                                       x not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]]

                    if len(src_non_special) >= 2 and len(tgt_non_special) >= 2:
                        processed_test.append({
                            "src_ids": torch.tensor(src_ids),
                            "tgt_ids": torch.tensor(tgt_ids)
                        })

                print(f"测试集 {year} 处理后样本: {len(processed_test)}")

                # 创建测试数据加载器
                test_loader = DataLoader(processed_test, batch_size=self.batch_size)
                test_loaders[year] = test_loader

            except Exception as e:
                print(f"加载测试集 {year} 时出错: {e}")
                continue

        return test_loaders