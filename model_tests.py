import numpy as np

def simulate_test():
    # Parâmetros fixos para o teste
    T_end = 10.0
    dt = 1.0
    t_eval = np.arange(0, T_end + dt, dt)

    results = {
        "t": t_eval,
        "H": np.random.random(len(t_eval)),
        "Df": np.random.random(len(t_eval)),
        "TempC": np.random.random(len(t_eval)) * 100 + 300,
        "x": np.random.random(len(t_eval)),
        "D_Pd": np.random.random(len(t_eval))
    }
    
    for key, value in results.items():
        print(f'{key}: {value}')

    return results

# Executa o teste básico
if __name__ == "__main__":
    simulate_test()