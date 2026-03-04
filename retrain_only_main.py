#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from federated_continual_learning_main import main


if __name__ == "__main__":
    if "--retrain_only" not in sys.argv:
        sys.argv.append("--retrain_only")
    main()
