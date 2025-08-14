# PromptTSS

Welcome to the official codebase of PromptTSS: A Prompting-Based Approach for Interactive Multi-Granularity Time Series Segmentation, accepted to CIKM 2025.

# Usage

1. Install Python 3.8, and use `requirements.txt` to install the dependencies

   ```
   pip install -r requirements.txt
   ```
2. To execute the script with configuration settings passed via argparse, use:

   ```
   python main.py --...
   ```

   Alternatively, if you prefer to use locally defined parameters to overwrite args for faster experimentation iterations, run:
   ```
   python main.py --overwrite_args
   ```

# Contact

If you have any questions or suggestions, please reach out to Ching Chang at [blacksnail789521@gmail.com](mailto:blacksnail789521@gmail.com), or raise them in the 'Issues' section.

# Acknowledgement

This library was built upon the following repositories:

* TimeDRL: [https://github.com/blacksnail789521/TimeDRL](https://github.com/blacksnail789521/TimeDRL)
