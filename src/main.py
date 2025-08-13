import sys
from controller.verification_run import run_verification, run_standard
from controller.validation import run_validation


def choose_mode():
    print("Scegli la modalità:")
    print("1. Verifica (distribuzioni esponenziali)")
    print("2. Simulazione standard (distribuzioni iperesponenziali)")
    print("3. Validation (sweep ARRIVAL_P/L1/L2 e salvataggio CSV)")
    print("4. Esci")
    choice = input("Inserisci: ").strip()
    if choice == "1":
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
            run_verification()
        elif mode == "standard":
            run_standard()
        elif mode == "validation":
            run_validation()
        else:
            run_standard()


if __name__ == "__main__":
    run_sim()
