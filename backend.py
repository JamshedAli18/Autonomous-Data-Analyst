import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import json
import numpy as np

load_dotenv()

# ============================================
# Initialize ONLY the 3 Models You Specified
# ============================================

planner_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

code_gen_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

explanation_llm = ChatGroq(
    model="groq/compound-mini",
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

class DataAnalyzer:
    def __init__(self, csv_file_path):
        """Load CSV file"""
        self.df = pd.read_csv(csv_file_path)
        self.file_name = os.path.basename(csv_file_path)
        self.charts_generated = []
        
    def get_data_summary(self):
        """Get basic data statistics"""
        summary = {
            "rows": len(self.df),
            "columns": list(self.df.columns),
            "shape": self.df.shape,
            "data_types": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_columns": self.df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": self.df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": self.df.select_dtypes(include=['datetime64']).columns.tolist(),
            "basic_stats": self.df.describe().to_dict()
        }
        return summary
    
    # ============================================
    # CHART EXPLANATION GENERATOR
    # ============================================
    def explain_chart(self, chart_title, chart_type, column_info):
        """Generate simple explanation for a chart"""
        prompt_template = PromptTemplate(
            input_variables=["chart_title", "chart_type", "column_info"],
            template="""You are a data visualization expert. Explain this chart in simple, easy language:

Chart Title: {chart_title}
Chart Type: {chart_type}
Column Information: {column_info}

Provide a 1-2 sentence explanation that:
- Anyone can understand (no jargon)
- Explains what the chart shows
- Is concise and clear

Keep it to maximum 2 sentences."""
        )
        
        chain = LLMChain(llm=explanation_llm, prompt=prompt_template)
        explanation = chain.run(
            chart_title=chart_title,
            chart_type=chart_type,
            column_info=column_info
        )
        
        return explanation.strip()
    
    # ============================================
    # SMART COLUMN TYPE INSPECTION
    # ============================================
    def inspect_columns(self):
        """Inspect and categorize all columns"""
        info = {
            "numeric": self.df.select_dtypes(include=['number']).columns.tolist(),
            "categorical": self.df.select_dtypes(include=['object']).columns.tolist(),
            "datetime": self.df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Try to detect datetime columns that might be stored as strings
        for col in info["categorical"]:
            try:
                pd.to_datetime(self.df[col], errors='coerce')
                if self.df[col].apply(lambda x: isinstance(x, str) and len(x) in [10, 19]).sum() > len(self.df) * 0.8:
                    info["datetime"].append(col)
                    info["categorical"].remove(col)
            except:
                pass
        
        return info
    
    # ============================================
    # SMART CHART DECISION LOGIC
    # ============================================
    def decide_charts(self):
        """Decide which charts to generate based on data"""
        charts_to_create = []
        col_info = self.inspect_columns()
        
        # Rule 1: If numeric columns exist → Histogram (Distribution)
        if col_info["numeric"]:
            charts_to_create.append({
                "type": "histogram",
                "column": col_info["numeric"][0],
                "title": f"Distribution of {col_info['numeric'][0]}",
                "priority": 1
            })
        
        # Rule 2: If multiple numeric columns → Correlation heatmap
        if len(col_info["numeric"]) > 1:
            charts_to_create.append({
                "type": "correlation_heatmap",
                "columns": col_info["numeric"],
                "title": "Feature Correlation",
                "priority": 1
            })
            
            # Rule 3: If two strong correlated features → Scatter plot
            correlation = self.df[col_info["numeric"]].corr()
            strong_corr = []
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    if abs(correlation.iloc[i, j]) > 0.5:
                        col1 = correlation.columns[i]
                        col2 = correlation.columns[j]
                        strong_corr.append((col1, col2, correlation.iloc[i, j]))
            
            if strong_corr:
                top_corr = sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)[0]
                charts_to_create.append({
                    "type": "scatter",
                    "col1": top_corr[0],
                    "col2": top_corr[1],
                    "title": f"{top_corr[0]} vs {top_corr[1]} (Correlation: {top_corr[2]:.2f})",
                    "priority": 2
                })
        
        # Rule 4: If categorical column exists → Bar chart
        if col_info["categorical"]:
            best_cat = col_info["categorical"][0]
            if self.df[best_cat].nunique() < 20:
                charts_to_create.append({
                    "type": "bar",
                    "column": best_cat,
                    "title": f"Distribution by {best_cat}",
                    "priority": 2
                })
        
        # Rule 5: If date/time column exists → Line chart
        if col_info["datetime"]:
            date_col = col_info["datetime"][0]
            if col_info["numeric"]:
                charts_to_create.append({
                    "type": "line",
                    "date_col": date_col,
                    "value_col": col_info["numeric"][0],
                    "title": f"{col_info['numeric'][0]} Trend Over Time",
                    "priority": 2
                })
        
        # Rule 6: Box plot for outlier detection
        if col_info["numeric"]:
            charts_to_create.append({
                "type": "boxplot",
                "columns": col_info["numeric"][:3],
                "title": "Outlier Detection (Box Plot)",
                "priority": 3
            })
        
        charts_to_create.sort(key=lambda x: x["priority"])
        return charts_to_create[:6]
    
    # ============================================
    # CHART GENERATION METHODS
    # ============================================
    def create_histogram(self, column, title, output_dir="uploads"):
        """Create histogram for numeric column"""
        try:
            plt.figure(figsize=(8, 5))
            plt.hist(self.df[column].dropna(), bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3, axis='y')
            
            file_path = os.path.join(output_dir, f"chart_histogram_{column.replace(' ', '_')}.png")
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Generate explanation
            explanation = self.explain_chart(
                title,
                "Histogram",
                f"Shows distribution of {column} values"
            )
            
            self.charts_generated.append({
                "file": file_path,
                "title": title,
                "explanation": explanation
            })
            return file_path
        except Exception as e:
            print(f"Error creating histogram: {e}")
            return None
    
    def create_correlation_heatmap(self, columns, title, output_dir="uploads"):
        """Create correlation heatmap"""
        try:
            plt.figure(figsize=(10, 8))
            correlation = self.df[columns].corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title(title, fontsize=14, fontweight='bold')
            
            file_path = os.path.join(output_dir, "chart_correlation_heatmap.png")
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Generate explanation
            explanation = self.explain_chart(
                title,
                "Heatmap",
                f"Shows relationships between columns: {', '.join(columns)}"
            )
            
            self.charts_generated.append({
                "file": file_path,
                "title": title,
                "explanation": explanation
            })
            return file_path
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None
    
    def create_scatter(self, col1, col2, title, output_dir="uploads"):
        """Create scatter plot for two numeric columns"""
        try:
            plt.figure(figsize=(8, 5))
            plt.scatter(self.df[col1], self.df[col2], alpha=0.6, color='#A23B72', s=50, edgecolors='black')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.grid(alpha=0.3)
            
            file_path = os.path.join(output_dir, f"chart_scatter_{col1}_{col2}.png")
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Generate explanation
            explanation = self.explain_chart(
                title,
                "Scatter Plot",
                f"Compares {col1} and {col2} values"
            )
            
            self.charts_generated.append({
                "file": file_path,
                "title": title,
                "explanation": explanation
            })
            return file_path
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            return None
    
    def create_bar(self, column, title, output_dir="uploads"):
        """Create bar chart for categorical column"""
        try:
            plt.figure(figsize=(10, 5))
            value_counts = self.df[column].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values, color='#F18F01', edgecolor='black', alpha=0.7)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.ylabel('Count')
            plt.grid(alpha=0.3, axis='y')
            
            file_path = os.path.join(output_dir, f"chart_bar_{column.replace(' ', '_')}.png")
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Generate explanation
            explanation = self.explain_chart(
                title,
                "Bar Chart",
                f"Counts different categories in {column}"
            )
            
            self.charts_generated.append({
                "file": file_path,
                "title": title,
                "explanation": explanation
            })
            return file_path
        except Exception as e:
            print(f"Error creating bar chart: {e}")
            return None
    
    def create_line(self, date_col, value_col, title, output_dir="uploads"):
        """Create line chart for time series"""
        try:
            df_sorted = self.df.sort_values(by=date_col)
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
            
            plt.figure(figsize=(10, 5))
            plt.plot(df_sorted[date_col], df_sorted[value_col], marker='o', color='#06A77D', linewidth=2, markersize=4)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel(value_col)
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            
            file_path = os.path.join(output_dir, f"chart_line_{value_col}.png")
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Generate explanation
            explanation = self.explain_chart(
                title,
                "Line Chart",
                f"Shows how {value_col} changes over time"
            )
            
            self.charts_generated.append({
                "file": file_path,
                "title": title,
                "explanation": explanation
            })
            return file_path
        except Exception as e:
            print(f"Error creating line chart: {e}")
            return None
    
    def create_boxplot(self, columns, title, output_dir="uploads"):
        """Create box plot for outlier detection"""
        try:
            plt.figure(figsize=(10, 5))
            self.df[columns].boxplot()
            plt.title(title, fontsize=14, fontweight='bold')
            plt.ylabel('Values')
            plt.grid(alpha=0.3, axis='y')
            
            file_path = os.path.join(output_dir, "chart_boxplot_outliers.png")
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Generate explanation
            explanation = self.explain_chart(
                title,
                "Box Plot",
                f"Identifies unusual or extreme values in data"
            )
            
            self.charts_generated.append({
                "file": file_path,
                "title": title,
                "explanation": explanation
            })
            return file_path
        except Exception as e:
            print(f"Error creating box plot: {e}")
            return None
    
    # ============================================
    # SMART CHART GENERATION
    # ============================================
    def generate_smart_charts(self, output_dir="uploads"):
        """Intelligently generate 3-6 best charts"""
        os.makedirs(output_dir, exist_ok=True)
        
        charts_to_create = self.decide_charts()
        
        print(f"\n Generating {len(charts_to_create)} smart charts...")
        
        for chart in charts_to_create:
            if chart["type"] == "histogram":
                self.create_histogram(chart["column"], chart["title"], output_dir)
            
            elif chart["type"] == "correlation_heatmap":
                self.create_correlation_heatmap(chart["columns"], chart["title"], output_dir)
            
            elif chart["type"] == "scatter":
                self.create_scatter(chart["col1"], chart["col2"], chart["title"], output_dir)
            
            elif chart["type"] == "bar":
                self.create_bar(chart["column"], chart["title"], output_dir)
            
            elif chart["type"] == "line":
                self.create_line(chart["date_col"], chart["value_col"], chart["title"], output_dir)
            
            elif chart["type"] == "boxplot":
                self.create_boxplot(chart["columns"], chart["title"], output_dir)
        
        return [chart["file"] for chart in self.charts_generated]
    
    # ============================================
    # STEP 1: Planner (llama-3.3-70b-versatile)
    # ============================================
    def plan_analysis(self):
        """Use Planner Model for reasoning"""
        summary = self.get_data_summary()
        
        prompt_template = PromptTemplate(
            input_variables=["data_info"],
            template="""You are a Data Analysis Planner. Analyze this dataset and create a plan:

Dataset Info:
{data_info}

Create a brief analysis plan with:
1. Main data characteristics
2. Potential insights to extract
3. Best visualizations for this data
4. Key patterns to look for

Be concise and specific to THIS dataset."""
        )
        
        chain = LLMChain(llm=planner_llm, prompt=prompt_template)
        plan = chain.run(data_info=json.dumps(summary, indent=2, default=str))
        
        return plan
    
    # ============================================
    # STEP 2: Code Generator (openai/gpt-oss-120b)
    # ============================================
    def generate_analysis_code(self):
        """Use Code Generation Model"""
        summary = self.get_data_summary()
        
        numeric_cols = summary['numeric_columns']
        categorical_cols = summary['categorical_columns']
        
        numeric_str = ", ".join(numeric_cols) if numeric_cols else "None"
        categorical_str = ", ".join(categorical_cols) if categorical_cols else "None"
        
        prompt_template = PromptTemplate(
            input_variables=["numeric_cols", "categorical_cols"],
            template="""You are a Python Data Analysis Code Generator.

Generate Python code (using pandas & numpy) to analyze this data:
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

Generate code snippets for:
1. Statistical analysis
2. Outlier detection
3. Correlation analysis
4. Basic trends

Format: Return only working Python code, no explanations."""
        )
        
        chain = LLMChain(llm=code_gen_llm, prompt=prompt_template)
        code = chain.run(numeric_cols=numeric_str, categorical_cols=categorical_str)
        
        return code
    
    # ============================================
    # STEP 3: Explanation Model (groq/compound-mini)
    # ============================================
    def generate_insights(self):
        """Use Explanation Model - Returns insights as bullet points"""
        summary = self.get_data_summary()
        
        prompt_template = PromptTemplate(
            input_variables=["data_summary"],
            template="""You are a Data Insight Explainer. Extract key insights from this data.

Data Summary:
{data_summary}

Provide EXACTLY 5 key insights in this format (use dashes for bullet points):
- First insight about the data
- Second insight about patterns
- Third insight about relationships
- Fourth insight about distributions
- Fifth insight about anomalies or special findings

Requirements:
- Use simple English (8th grade level)
- No technical jargon
- Be specific to the actual data
- Start each line with a dash (-)
- Keep each insight to 1-2 sentences max"""
        )
        
        chain = LLMChain(llm=explanation_llm, prompt=prompt_template)
        insights_text = chain.run(data_summary=json.dumps(summary, indent=2, default=str))
        
        lines = insights_text.strip().split('\n')
        clean_insights = []
        
        for line in lines:
            line = line.strip()
            if line:
                if not line.startswith('-'):
                    line = f"- {line}"
                clean_insights.append(line)
        
        return '\n'.join(clean_insights[:5])
    
    # ============================================
    # Main Analysis Workflow
    # ============================================
    def analyze(self):
        """
        Run complete analysis using 3 specialized models
        + smart chart generation with explanations
        """
        result = {
            "file_name": self.file_name,
            "summary": self.get_data_summary(),
            "analysis_plan": self.plan_analysis(),
            "generated_code": self.generate_analysis_code(),
            "insights": self.generate_insights(),
            "charts": self.generate_smart_charts(),
            "chart_titles": [c["title"] for c in self.charts_generated],
            "chart_explanations": [c["explanation"] for c in self.charts_generated]
        }
        return result


def test_analyzer(csv_path):
    """Test the analyzer"""
    analyzer = DataAnalyzer(csv_path)
    result = analyzer.analyze()
    print("Analysis Complete!")

    return result
