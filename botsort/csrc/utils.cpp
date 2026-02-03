#include "include/utils.h"

#include <iostream>

#include "include/lapjv.h"

double lapjv(
    CostMatrix &cost, 
    std::vector<int> &rowsol,
    std::vector<int> &colsol, 
    bool extend_cost, 
    float cost_limit,
    bool return_cost)
{

    int n_rows = static_cast<int>(cost.rows());
    int n_cols = static_cast<int>(cost.cols());
    int n = n_rows;
    double fill_value = -1;

    if (!extend_cost && (n_rows != n_cols))
    {
        throw std::runtime_error("Unequal Rows to Columns. Set extend_cost=True");
    }

    if ((cost_limit < LONG_MAX) || extend_cost)
    {
        n = n_rows + n_cols;
        if (cost_limit < LONG_MAX)
        {
            fill_value = cost_limit / 2.0f;
        }
        else{
            fill_value = cost.maxCoeff();
            fill_value++;
        }
    }

    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    double **cost_ptr = new double* [n];
    for (int i = 0; i < n_rows; i++)
    {  
        cost_ptr[i] = new double[n];
        for (int j = 0; j < n_cols; j++)//Q1 real values
        {   
            cost_ptr[i][j] = static_cast<double>(cost(i, j));
        }
        for (int j = n_cols; j < n; j++)//Q2 fill values
        {
            cost_ptr[i][j] = fill_value;
        }
    }
    for (int i = n_rows; i < n; i++)
    {
        cost_ptr[i] = new double[n];
        for (int j = 0; j < n_cols; j++)//Q3 real values
        {   
            cost_ptr[i][j] = fill_value;
        }
        for (int j = n_cols; j < n; j++)//Q4 fill values
        {
            cost_ptr[i][j] = 0;
        }
    }

    int *x_c = new int[n];
    int *y_c = new int[n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0)
    {
        for (int i = 0; i < n; i++) { delete[] cost_ptr[i]; }
        delete[] cost_ptr;
        delete[] x_c;
        delete[] y_c;
        throw std::runtime_error("LAPJV algorithm failed");
    }

    if (n != n_rows) {
        for (int i = 0; i < n_rows; ++i)
        {
            if (x_c[i] >= n_cols) 
                rowsol[i] = -1;
            else
                rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; ++i)
        {
            if (y_c[i] >= n_rows) 
                colsol[i] = -1;
            else
                colsol[i] = y_c[i];
        }
    }

    double opt = 0.0;
    if (return_cost)
    {
        for (size_t i = 0; i < rowsol.size(); i++)
        {
            if (rowsol[i] >= 0)
                {opt += cost_ptr[i][rowsol[i]];}
        }
    }

    for (int i = 0; i < n; i++) { delete[] cost_ptr[i]; }
    delete[] cost_ptr;
    delete[] x_c;
    delete[] y_c;

    return opt;
}
