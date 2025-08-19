import sys
from controller.simulation import run_simulation, run_finite_horizon, run_infinite_horizon
from engineering.costants import *

from library.rngs import plantSeeds
from engineering.costants import RNG_SEED_VERIFICATION, RNG_SEED_STANDARD, BURN_IN



def choose_mode():
    print("Scegli la modalità:")
    print("0. Singola run")
    print("1. Verifica (distribuzioni esponenziali)")
    print("2. Simulazione standard (distribuzioni iperesponenziali)")
    print("3. Validazione")
    print("4. Analisi del Transitorio")
    print("5. Analisi ad Orizzonte Finito")
    print("6. Analisi ad Orizzonte Infinito")
    print("7. Esci")
    choice = input("Inserisci: ").strip()
    if choice == "0":
        return "single" 
    elif choice == "1":
        return "verification"
    elif choice == "2":
        return "standard"
    elif choice == "3":
        return "validation"
    elif choice == "4":
        return "transitory"
    elif choice == "5":
        return "finite simulation"
    elif choice == "6":
        return "infinite simulation"
    elif choice == "7":
        sys.exit()
    # else:
    #     print("Scelta non valida. Default: standard.")
    #     return "standard"


def run_sim():
    
    print()
    while True:
        mode = choose_mode()
        print(f"\nModalità selezionata: {mode.upper()}")

        if mode == "verification":
            plantSeeds(RNG_SEED_VERIFICATION)
        else:
            plantSeeds(RNG_SEED_STANDARD)

        if mode == "verification":

            run_simulation("x1", "verification", enable_windowing=True)

        elif mode == "standard":

            run_simulation("x1","standard", enable_windowing=True)

        elif mode == "single":

            run_simulation("x1","standard", enable_windowing=False)

        elif mode == "validation":

            run_simulation("x1","standard", enable_windowing=True)
            run_simulation("x2", "standard", enable_windowing=True, arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x2, arrival_l2=ARRIVAL_L2_x2)
            run_simulation("x5","standard", enable_windowing=True, arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x5, arrival_l2=ARRIVAL_L2_x5)
            run_simulation("x10","standard", enable_windowing=True, arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x10, arrival_l2=ARRIVAL_L2_x10)
            run_simulation("x40","standard", enable_windowing=True, arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x40, arrival_l2=ARRIVAL_L2_x40)

        elif mode == "transitory":

            logs = run_finite_horizon("transitory", scenario="transitory_x40",
                              out_csv="plot/results_transitory.csv")
            
            # for r in logs[:5]:
            #     print(r)
            
            print(f"\n[OK] Log del transitorio salvati in: plot/results_transitory.csv")

        elif mode == "finite simulation":

            logs = run_finite_horizon("finite simulation", enable_windowing=False,
                              scenario="finite simulation",
                              out_csv="plot/results_finite_simulation.csv")
            
            # for r in logs[:5]:
            #     print(r)
            
            print(f"\n[OK] Log salvati in: plot/results_finite_simulation.csv")

        elif mode == "infinite simulation":
            run_infinite_horizon(
                mode="standard",        
                out_csv="plot/results_infinite_bm.csv",
                out_acs="acs_input/acs_input.csv",
                burn_in=BURN_IN
            )


if __name__ == "__main__":
    run_sim()
