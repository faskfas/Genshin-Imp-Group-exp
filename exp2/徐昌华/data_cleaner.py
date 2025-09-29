#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class UniversalDataCleaner:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.original_df = None
        self.cleaning_log = []
        
    def log_action(self, action: str):
        """记录清洗操作"""
        print(f" {action}")
        self.cleaning_log.append(action)
        
    def load_data(self):
        """智能加载CSV数据，自动检测编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.input_file, encoding=encoding)
                self.original_df = self.df.copy()
                print(f" 成功加载数据（编码：{encoding}）")
                print(f"  - 数据维度：{self.df.shape[0]}行 × {self.df.shape[1]}列")
                print(f"  - 列名：{list(self.df.columns)}")
                return True
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"  尝试编码 {encoding} 失败：{e}")
                continue
        
        print(" 所有编码尝试失败！")
        return False
        
    def auto_detect_column_types(self):
        """自动检测列的数据类型"""
        print("\n=== 自动检测数据类型 ===")
        
        column_types = {}
        
        for col in self.df.columns:
            # 获取非空值
            non_null_values = self.df[col].dropna()
            
            if len(non_null_values) == 0:
                column_types[col] = 'empty'
                continue
                
            # 检查是否为数值型
            numeric_values = pd.to_numeric(non_null_values, errors='coerce')
            numeric_ratio = numeric_values.notna().sum() / len(non_null_values)
            
            if numeric_ratio > 0.8:  # 80%以上是数值
                if numeric_values.equals(numeric_values.astype(int, errors='ignore')):
                    column_types[col] = 'integer'
                else:
                    column_types[col] = 'float'
            else:
                # 检查是否为布尔型
                bool_values = non_null_values.astype(str).str.upper()
                bool_keywords = {'TRUE', 'FALSE', 'YES', 'NO', '1', '0', 'T', 'F', 'Y', 'N'}
                bool_ratio = bool_values.isin(bool_keywords).sum() / len(non_null_values)
                
                if bool_ratio > 0.8:
                    column_types[col] = 'boolean'
                else:
                    # 检查是否为分类型（重复值较多）
                    unique_ratio = len(non_null_values.unique()) / len(non_null_values)
                    if unique_ratio < 0.1:  # 唯一值比例小于10%
                        column_types[col] = 'category'
                    else:
                        column_types[col] = 'text'
            
            print(f"  {col}: {column_types[col]}")
            
        return column_types
        
    def detect_and_fix_missing_values(self):
        """检测和处理缺失值"""
        print("\n=== 处理缺失值 ===")
        
        missing_info = self.df.isnull().sum()
        total_missing = missing_info.sum()
        
        if total_missing == 0:
            print(" 未发现缺失值")
            return
            
        print(f"发现 {total_missing} 个缺失值：")
        
        for col, count in missing_info[missing_info > 0].items():
            missing_ratio = count / len(self.df)
            print(f"  {col}: {count}个 ({missing_ratio:.2%})")
            
            if missing_ratio > 0.5:  # 超过50%缺失，考虑删除列
                print(f"    警告：{col}列缺失率过高，建议删除")
                # 可以选择删除该列
                # self.df = self.df.drop(columns=[col])
                # self.log_action(f"删除缺失率过高的列：{col}")
            else:
                # 根据数据类型填充缺失值
                if self.df[col].dtype in ['int64', 'float64']:
                    # 数值型：用中位数填充
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    self.log_action(f"用中位数({median_val})填充{col}列的缺失值")
                else:
                    # 文本型：用众数填充
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                        self.log_action(f"用众数({mode_val[0]})填充{col}列的缺失值")
        
    def detect_and_remove_duplicates(self):
        """检测和删除重复行"""
        print("\n=== 处理重复数据 ===")
        
        initial_rows = len(self.df)
        duplicates = self.df.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            print(f"发现 {duplicate_count} 个重复行")
            
            # 显示重复行的示例
            duplicate_rows = self.df[self.df.duplicated(keep=False)]
            print("重复行示例（前5行）：")
            print(duplicate_rows.head())
            
            # 删除重复行
            self.df = self.df.drop_duplicates()
            removed_rows = initial_rows - len(self.df)
            self.log_action(f"删除了{removed_rows}个重复行")
        else:
            print(" 未发现重复行")
            
    def detect_and_fix_outliers(self):
        """检测和处理异常值"""
        print("\n=== 处理异常值 ===")
        
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            print(" 无数值列，跳过异常值检测")
            return
            
        outlier_info = {}
        
        for col in numeric_columns:
            # 使用IQR方法检测异常值
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_ratio = len(outliers) / len(self.df)
                outlier_info[col] = {
                    'count': len(outliers),
                    'ratio': outlier_ratio,
                    'bounds': (lower_bound, upper_bound)
                }
                
                print(f"  {col}: {len(outliers)}个异常值 ({outlier_ratio:.2%})")
                print(f"    正常范围：[{lower_bound:.2f}, {upper_bound:.2f}]")
                
                # 如果异常值比例不高，可以考虑用边界值替换
                if outlier_ratio < 0.05:  # 小于5%
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                    self.log_action(f"将{col}列的异常值限制在正常范围内")
        
        if not outlier_info:
            print(" 未发现显著异常值")
            
    def standardize_data_types(self):
        """标准化数据类型"""
        print("\n=== 标准化数据类型 ===")
        
        column_types = self.auto_detect_column_types()
        
        for col, dtype in column_types.items():
            try:
                if dtype == 'integer':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')
                    self.log_action(f"将{col}列转换为整数类型")
                    
                elif dtype == 'float':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    self.log_action(f"将{col}列转换为浮点数类型")
                    
                elif dtype == 'boolean':
                    # 标准化布尔值
                    bool_map = {
                        'TRUE': True, 'FALSE': False,
                        'YES': True, 'NO': False,
                        'Y': True, 'N': False,
                        'T': True, 'F': False,
                        '1': True, '0': False,
                        'true': True, 'false': False,
                        'yes': True, 'no': False
                    }
                    self.df[col] = self.df[col].astype(str).map(bool_map)
                    self.log_action(f"将{col}列标准化为布尔类型")
                    
                elif dtype == 'category':
                    self.df[col] = self.df[col].astype('category')
                    self.log_action(f"将{col}列转换为分类类型")
                    
                elif dtype == 'text':
                    # 清理文本数据
                    self.df[col] = self.df[col].astype(str).str.strip()
                    # 移除多余的空格
                    self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
                    self.log_action(f"清理{col}列的文本数据")
                    
            except Exception as e:
                print(f"    警告：{col}列类型转换失败：{e}")
                
    def clean_text_data(self):
        """清理文本数据"""
        print("\n=== 清理文本数据 ===")
        
        text_columns = self.df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            original_values = self.df[col].copy()
            
            # 统一处理字符串
            self.df[col] = self.df[col].astype(str)
            
            # 移除首尾空格
            self.df[col] = self.df[col].str.strip()
            
            # 统一多个空格为单个空格
            self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
            
            # 移除或替换特殊字符（保留常见标点）
            self.df[col] = self.df[col].str.replace(r'[^\w\s\-\.\(\)♀♂°′″]', '', regex=True)
            
            # 检查是否有改动
            if not self.df[col].equals(original_values):
                self.log_action(f"清理了{col}列的文本格式")
                
    def validate_data_consistency(self):
        """验证数据一致性"""
        print("\n=== 验证数据一致性 ===")
        
        # 检查数值列的合理性
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col.lower() in ['total', '总计', 'sum']:
                # 如果是汇总列，检查是否等于其他列的和
                other_numeric_cols = [c for c in numeric_columns if c != col and 
                                    not any(keyword in c.lower() for keyword in ['total', 'sum', '总计', 'id', '#'])]
                
                if len(other_numeric_cols) > 1:
                    calculated_total = self.df[other_numeric_cols].sum(axis=1)
                    current_total = self.df[col]
                    
                    # 检查差异
                    diff = abs(calculated_total - current_total)
                    mismatched = diff > 1  # 容忍1的误差
                    
                    if mismatched.sum() > 0:
                        print(f"  发现{mismatched.sum()}行{col}列数值不一致")
                        # 可以选择修正
                        # self.df[col] = calculated_total
                        # self.log_action(f"修正了{col}列的计算值")
                        
        print("✓ 数据一致性检查完成")
        
    def generate_summary_report(self):
        """生成清洗摘要报告"""
        print("\n" + "="*60)
        print("           数据清洗摘要报告")
        print("="*60)
        
        print(f" 输入文件：{self.input_file}")
        print(f" 输出文件：{self.output_file}")
        print(f" 原始数据：{self.original_df.shape[0]}行 × {self.original_df.shape[1]}列")
        print(f" 清洗后：{self.df.shape[0]}行 × {self.df.shape[1]}列")
        
        # 数据变化统计
        rows_changed = self.original_df.shape[0] - self.df.shape[0]
        if rows_changed != 0:
            print(f" 行数变化：{rows_changed:+d}")
            
        print(f"\n🔧 执行的清洗操作（{len(self.cleaning_log)}项）：")
        for i, action in enumerate(self.cleaning_log, 1):
            print(f"   {i}. {action}")
            
        # 数据质量指标
        print(f"\n 清洗后数据质量：")
        print(f"   - 缺失值：{self.df.isnull().sum().sum()}")
        print(f"   - 重复行：{self.df.duplicated().sum()}")
        print(f"   - 数据类型：{len(self.df.dtypes.unique())}种")
        
        # 各列的数据类型分布
        type_counts = self.df.dtypes.value_counts()
        print(f"\n 数据类型分布：")
        for dtype, count in type_counts.items():
            print(f"   - {dtype}: {count}列")
            
    def save_cleaned_data(self):
        """保存清洗后的数据"""
        try:
            self.df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
            print(f"\n 清洗后的数据已保存到：{self.output_file}")
            return True
        except Exception as e:
            print(f" 保存数据失败：{e}")
            return False
            
    def run_cleaning_pipeline(self):
        """执行完整的数据清洗流程"""
        print(" 启动通用自动化数据清洗流程")
        print("="*60)
        
        # 1. 加载数据
        if not self.load_data():
            return False
            
        # 2. 自动检测数据类型
        self.auto_detect_column_types()
        
        # 3. 处理缺失值
        self.detect_and_fix_missing_values()
        
        # 4. 处理重复数据
        self.detect_and_remove_duplicates()
        
        # 5. 处理异常值
        self.detect_and_fix_outliers()
        
        # 6. 标准化数据类型
        self.standardize_data_types()
        
        # 7. 清理文本数据
        self.clean_text_data()
        
        # 8. 验证数据一致性
        self.validate_data_consistency()
        
        # 9. 保存清洗后的数据
        if self.save_cleaned_data():
            # 10. 生成报告
            self.generate_summary_report()
            return True
        
        return False

def main():
    """主函数"""
    input_file = "e:/bigdatapractice/lab2/Pokemon.csv"
    output_file = "e:/bigdatapractice/lab2/Pokemon_cleaned.csv"
    
    print(" 通用自动化数据清洗工具")
    print("适用于任何CSV数据集，无需事先了解数据内容")
    print("="*60)
    
    # 创建清洗器实例
    cleaner = UniversalDataCleaner(input_file, output_file)
    
    # 执行清洗流程
    if cleaner.run_cleaning_pipeline():
        print("\n 数据清洗已完成！")
    else:
        print("\n try again！")

if __name__ == "__main__":
    main()