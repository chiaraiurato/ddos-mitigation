import sys
from controller.verification_run import run_simulation
from controller.validation import run_validation

from library.rngs import plantSeeds
from engineering.costants import RNG_SEED_VERIFICATION, RNG_SEED_STANDARD



def choose_mode():
    print("Scegli la modalità:")
    print("0. Singola run")
    print("1. Verifica (distribuzioni esponenziali)")
    print("2. Simulazione standard (distribuzioni iperesponenziali)")
    print("3. Validation (sweep ARRIVAL_P/L1/L2 e salvataggio CSV)")
    print("4. Esci")
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
        sys.exit()
    else:
        print("Scelta non valida. Default: standard.")
        return "standard"


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
            run_simulation("verification", batch_means=True)
        elif mode == "standard":
            run_simulation("standard", batch_means=True)
        elif mode == "single":
            run_simulation("standard", batch_means=False)
        # elif mode == "validation":
        #     run_validation()
        # else:
        #     run_standard()


if __name__ == "__main__":
    run_sim()
