import sys
from controller.simulation import run_simulation, run_finite_horizon, run_infinite_horizon, run_verification
from engineering.costants import *
from library.rngs import plantSeeds


def choose_variant():
    print_banner()
    print("Scegli il MODELLO:")
    print("A. Modello di BASE (solo Mitigation)")
    print("B. Modello MIGLIORATIVO (Mitigation + Analysis Center ML)")
    print("Exit. Per uscire")

    v = input("Inserisci [A/B/Exit]: ").strip().lower()
    if v == "exit":
        sys.exit()

    if v == "b":
        return "ml_analysis"
    return "baseline"


def choose_mode():

    print("\nScegli la modalità:")
    print("0. Simulazione senza attacco")
    print("1. Verifica (distribuzioni esponenziali)")
    print("2. Simulazione standard (distribuzioni iperesponenziali)")
    print("3. Validazione (x2, x5, x10, x40)")
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
    else:
        print("Scelta non valida. Default: standard.")
        return "standard"


def run_sim():
    print()
    while True:
        variant = choose_variant()
        print(f"\nModello selezionato: {'MIGLIORATIVO' if variant=='ml_analysis' else 'BASE'}")

        mode = choose_mode()
        print(f"Modalità selezionata: {mode.upper()}")

        if mode == "verification":
            plantSeeds(RNG_SEED_VERIFICATION)
        else:
            plantSeeds(RNG_SEED_STANDARD)

        if variant == 'baseline':
            model = 'baseline'
        else:
            model = 'ml_analysis'

        if mode == "verification":
            run_verification(model, enable_windowing=True)

        elif mode == "standard":
            run_simulation("x40", "standard", model, enable_windowing=True,
                           arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x40, arrival_l2=ARRIVAL_L2_x40)

        elif mode == "single":
            run_simulation("x1", "standard", model, enable_windowing=False)

        elif mode == "validation":
            run_simulation("x1",  "standard", model, enable_windowing=True)
            run_simulation("x2",  "standard", model, enable_windowing=True,
                           arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x2,  arrival_l2=ARRIVAL_L2_x2)
            run_simulation("x5",  "standard", model, enable_windowing=True,
                           arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x5,  arrival_l2=ARRIVAL_L2_x5)
            run_simulation("x10", "standard", model, enable_windowing=True,
                           arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x10, arrival_l2=ARRIVAL_L2_x10)
            run_simulation("x40", "standard", model, enable_windowing=True,
                           arrival_p=ARRIVAL_P, arrival_l1=ARRIVAL_L1_x40, arrival_l2=ARRIVAL_L2_x40)

        elif mode == "transitory":
            logs = run_finite_horizon("transitory", scenario="transitory_x40",
                                      out_csv="plot/results_transitory_" + model + ".csv", 
                                      model=model)
            
        elif mode == "finite simulation":
            logs = run_finite_horizon("finite simulation",
                                        scenario="finite simulation",
                                        out_csv="plot/results_finite_simulation_" + model + ".csv",
                                        model=model)
            
        elif mode == "infinite simulation":
            run_infinite_horizon(
                mode="standard",
                out_csv="plot/results_infinite_bm_" + model + ".csv",
                out_acs="acs_input/acs_input_ " + model + ".csv",
                burn_in=BURN_IN,
                model=model
            )
def print_banner():
    print(r"""
██████╗ ██████╗  ██████╗ ███████╗    ███╗   ███╗██╗████████╗██╗ ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔══██╗██╔═══██╗██╔════╝    ████╗ ████║██║╚══██╔══╝██║██╔════╝ ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
██║  ██║██║  ██║██║   ██║███████╗    ██╔████╔██║██║   ██║   ██║██║  ███╗███████║   ██║   ██║██║   ██║██╔██╗ ██║
██║  ██║██║  ██║██║   ██║╚════██║    ██║╚██╔╝██║██║   ██║   ██║██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
██████╔╝██████╔╝╚██████╔╝███████║    ██║ ╚═╝ ██║██║   ██║   ██║╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝    ╚═╝     ╚═╝╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
""", flush=True)
    
if __name__ == "__main__":
    run_sim()
