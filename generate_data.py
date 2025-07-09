import pandas as pd
import numpy as np

def generate_employee_data(num_records=250):
    """
    Generates a comprehensive employee dataset suitable for regression,
    clustering, and classification tasks, as well as rich visualizations.
    """
    np.random.seed(42)

    # Departments and Salary Levels
    departments = ['Engineering', 'Sales', 'Human Resources', 'Marketing', 'Support']
    salary_levels = ['Low', 'Medium', 'High']

    # Generate core employee data
    data = {
        'EmployeeID': [f'EMP_{1001 + i}' for i in range(num_records)],
        'SatisfactionScore': np.random.normal(0.65, 0.15, num_records).clip(0, 1).round(2),
        'ProjectCount': np.random.randint(2, 8, num_records),
        'AvgMonthlyHours': np.random.normal(200, 25, num_records).astype(int).clip(120, 300),
        'YearsAtCompany': np.random.randint(1, 10, num_records),
        'Department': np.random.choice(departments, num_records, p=[0.3, 0.25, 0.15, 0.15, 0.15]),
        'SalaryLevel': np.random.choice(salary_levels, num_records, p=[0.4, 0.4, 0.2])
    }

    df = pd.DataFrame(data)

    # --- Create Target Variables for ML Models ---

    # 1. PerformanceRating (for Linear Regression)
    # Influenced by satisfaction, projects, and hours, with some noise
    performance_score = (
        df['SatisfactionScore'] * 2
        + df['ProjectCount'] * 0.3
        - (df['AvgMonthlyHours'] - 200) / 50
        + df['YearsAtCompany'] * 0.1
    )
    df['PerformanceRating'] = (performance_score + np.random.normal(0, 0.2, num_records)).clip(1, 5).round(2)

    # 2. PromotedInLast2Years (for Logistic Regression)
    # Higher chance if performance is good, at company longer, and high project count
    promotion_chance = (
        df['PerformanceRating'] / 5 
        + df['YearsAtCompany'] / 10
        + df['ProjectCount'] / 10
    )
    # Use a logistic function (sigmoid) to create a probability
    prob = 1 / (1 + np.exp(-(promotion_chance - 0.9) * 4)) 
    df['PromotedInLast2Years'] = (np.random.rand(num_records) < prob).map({True: 'Yes', False: 'No'})

    return df

if __name__ == "__main__":
    # Generate the dataset
    employee_df = generate_employee_data(num_records=250)

    # Define the output filename
    output_filename = "employee_data.xlsx"

    # Save to Excel
    try:
        employee_df.to_excel(output_filename, index=False)
        print(f"✅ Successfully generated and saved '{output_filename}'")
        print("\nThis file contains a mix of data types and is ready for use in the application.")
        print("It includes:")
        print("  - Numerical columns for regression/clustering.")
        print("  - A binary 'PromotedInLast2Years' column for logistic regression.")
        print("  - Categorical columns for visualization and grouping.")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")

    # Display a preview
    print("\n📊 Data Preview:")
    print(employee_df.head()) 