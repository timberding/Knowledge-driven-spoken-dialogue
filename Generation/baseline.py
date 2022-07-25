import sys

if __name__ == "__main__":
    if '--generate' in sys.argv:
        from baseline_ch import generate as main
    else:
        from baseline_ch import main

    main.main()
