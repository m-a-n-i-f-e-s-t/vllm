#!/usr/bin/env python3
import sys
import re
import numpy as np
import os

def parse_val(s):
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def main():
    # Regex to capture: pid (x, y, z) idx (i1, i2...) name: val
    # Example: pid (38, 2, 0) idx ( 5) seq_pos: 5
    # Example: pid (53, 1, 0) idx () scheduled_tile_end: 2  (scalar)
    pattern = re.compile(r"pid\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*idx\s*\(([^)]*)\)\s*([^:]+):\s*(.*)")
    
    raw_data = {} # name -> pid -> list of (index_tuple, value)
    
    # Check if stdin is interactive
    if sys.stdin.isatty():
        print("Warning: Reading from stdin which is a TTY. Pipe output to this tool.")
        print("Example: python script.py | tdb")
        print("Press Ctrl+D to finish.")

    try:
        for line in sys.stdin:
            # Check for match
            match = pattern.search(line)
            if match:
                try:
                    # Parse PID
                    pid_x, pid_y, pid_z = map(int, match.group(1, 2, 3))
                    pid = (pid_x, pid_y, pid_z)
                    
                    # Parse IDX (can be empty for scalars)
                    idx_str = match.group(4).strip()
                    if idx_str:
                        idx_parts = [int(x.strip()) for x in idx_str.split(',')]
                        idx = tuple(idx_parts)
                    else:
                        idx = ()  # scalar value
                    
                    # Parse Name and Value
                    name = match.group(5).strip()
                    val_str = match.group(6).strip()
                    val = parse_val(val_str)
                    
                    if name not in raw_data:
                        raw_data[name] = {}
                    if pid not in raw_data[name]:
                        raw_data[name][pid] = []
                    
                    raw_data[name][pid].append((idx, val))
                except ValueError:
                    # Parsing failed (e.g. bad int conversion), treat as normal line
                    sys.stdout.write(line)
                    sys.stdout.flush()
            else:
                # Not a debug line, print it
                sys.stdout.write(line)
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass

    # Reconstruct tensors into a single high-dimensional array per variable
    # Shape: (max_pid_x+1, max_pid_y+1, max_pid_z+1, ...idx dims...)
    data = {}
    
    for name, pid_dict in raw_data.items():
        if not pid_dict:
            continue
        
        # Collect all items with full index (pid + idx)
        all_items = []  # list of (full_index, value)
        for pid, items in pid_dict.items():
            for idx, val in items:
                full_idx = pid + idx  # concatenate pid tuple with idx tuple
                all_items.append((full_idx, val))
        
        if not all_items:
            continue
        
        # Determine total dimensionality (3 for pid + idx dims)
        ndim = len(all_items[0][0])
        
        # Determine shape from max indices
        max_indices = [0] * ndim
        for full_idx, _ in all_items:
            for i, dim_idx in enumerate(full_idx):
                max_indices[i] = max(max_indices[i], dim_idx)
        
        shape = tuple(m + 1 for m in max_indices)
        
        # Determine dtype
        first_val = all_items[0][1]
        if isinstance(first_val, (int, float)):
            dtype = float
            fill_value = np.nan
        else:
            dtype = object
            fill_value = None
        
        arr = np.full(shape, fill_value, dtype=dtype)
        
        for full_idx, val in all_items:
            try:
                arr[full_idx] = val
            except IndexError:
                pass
        
        data[name] = arr

    print(f"\nCaptured {len(data)} variables.")
    for k, v in data.items():
        print(f"  - {k}: shape {v.shape}")
        
    print("Variables are available as local variables (e.g. 'seq_pos').")
    
    # Build namespace for interactive shell
    namespace = {'np': np, 'data': data}
    namespace.update(data)
    
    # Reopen stdin to allow interaction if input was piped
    if not sys.stdin.isatty():
        try:
            # Close the old stdin and reopen fd 0 from /dev/tty
            sys.stdin.close()
            tty_fd = os.open('/dev/tty', os.O_RDWR)
            if tty_fd != 0:
                os.dup2(tty_fd, 0)
                os.close(tty_fd)
            sys.stdin = open(0, 'r')
        except Exception as e:
            print(f"Warning: Could not reconnect to terminal: {e}")
            return

    # Try IPython first (best experience), fall back to code.interact
    try:
        from IPython import embed
        print("Welcome to tdb (IPython).")
        embed(user_ns=namespace, colors='neutral')
    except ImportError:
        import code
        import readline
        import rlcompleter
        
        # Enable tab completion
        readline.set_completer(rlcompleter.Completer(namespace).complete)
        readline.parse_and_bind("tab: complete")
        
        print("Welcome to tdb.")
        code.interact(banner='', local=namespace)

if __name__ == "__main__":
    main()
