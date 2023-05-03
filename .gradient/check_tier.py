
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os

hostname = os.getenv("HOSTNAME", "unknown")

# Free tier hosts
free_hostnames = [f"lr17-1-poplar-{i}" for i in range(1, 36)]
free_hostnames.append("lr17-1-poplar-63")
free_hostnames.append("lr17-1-poplar-64")

if hostname in free_hostnames:
    print("FREE")
else:
    print("PAID")
