__author__ = 'Lukáš Bartůněk'

def load_trained():
    with open('data/recommended_parameters.txt', 'r') as file:
        data = file.readline()
        data = data.split("|")
        q_t,s_t,t_weight,percentage = data
    return float(q_t),float(s_t),float(t_weight),float(percentage)