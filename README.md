# Project Title

Classification with Random Forest over bank marketing dataset

## Description

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were
based on phone calls. Often, more than one contact to the same client was required, in order to access if the product
(bank term deposit) would be (or not) subscribed. The classification goal is to predict if the client will subscribe
a term deposit (variable y).

Below is input and output variables descriptions.
Input variables:
1 - age (numeric)
2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
4 - education (categorical: "unknown","secondary","primary","tertiary")
5 - default: has credit in default? (binary: "yes","no")
6 - balance: average yearly balance, in euros (numeric) 
7 - housing: has housing loan? (binary: "yes","no")
8 - loan: has personal loan? (binary: "yes","no")
9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
10 - day: last contact day of the month (numeric)
11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
12 - duration: last contact duration, in seconds (numeric)
13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 
means client was not previously contacted)
15 - previous: number of contacts performed before this campaign and for this client (numeric)
16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

Output variable:
y - has the client subscribed a term deposit? (binary: "yes","no")

Before doing classification, some data cleaning and exploration analysis is done to have better insight about data.
For classification, in addition to normal train-test session over data, some hyperparamter optimization with search
grid is done to find out whether there could be any improvements.

## Getting Started

### Dependencies

* Code is written by jupyter notebook on Win10
* Required libraries are listed in first code segment

### Installing

* To have a copy of code and other materials you can clone at: https://github.com/mrezaa/water_treatment_data_science.git](https://github.com/mrezaa/classification-with-random-forest.git

### Executing program

* In jupyter notebook every code segment is easily executed with shift+Enter
* In other python interpreters you can use .py version

## Help

For any kinds of issues, feel free to contact me via email
## Authors

Mohammadreza Asadi

mrrasadi@gmail.com
