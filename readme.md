# Assignment 1 : MPI All-to-All Exchange Algorithms

This project implements and compares **three different all-to-all communication strategies** in MPI — **pairwise XOR exchange**, **linear exchange**, and **Torsten Hoefler’s bandwidth-optimal exchange** — against the standard `MPI_Alltoall` collective.

---

## Overview

Each MPI process starts with a send buffer:

```
sendbuf[i] = rank * 100 + i
```

Every process should end up receiving one value from every other process:

```
recvbuf[i] = i * 100 + rank
```

The goal is to implement this exchange using different algorithms and verify correctness by comparing results with `MPI_Alltoall`.

---

## Algorithms Implemented

### 1. **Pairwise XOR Exchange**

- Each process communicates with its partner computed as:
  [
  \text{partner} = \text{rank} \oplus \text{phase}
  ]
- Requires the number of processes (`size`) to be a **power of two**.
- In each phase, processes exchange one integer and synchronize with a barrier.
- **Complexity:** ( O(\log P) ) communication rounds.
- **Limitation:** Invalid for non–power-of-two process counts.

---

### 2. **Linear Exchange**

- Each process sends data to every other process in a sequential manner.
- Partners are chosen by cyclic shifting:
  [
  \text{send_to} = (\text{rank} + \text{step}) \bmod P
  ]
  [
  \text{recv_from} = (\text{rank} - \text{step} + P) \bmod P
  ]
- Uses `MPI_Sendrecv` for simultaneous send/receive per step.
- **Complexity:** ( O(P) )
- **Advantage:** Works for any number of processes.
- **Drawback:** Slow for large `P`.

---

### 3. **Torsten Hoefler Exchange (Bandwidth-Optimal)**

- Inspired by Hoefler’s optimized `MPI_Alltoall` algorithm.
- Divides processes into **subtrees** and performs multi-phase sends/receives using non-blocking communication (`MPI_Isend`/`MPI_Irecv`).
- Exploits communication parallelism for better bandwidth utilization.
- **Complexity:** ( O(P) ) but with reduced contention and overlapping communication.
- Includes synchronization barriers for fairness and verification.

---

## Verification and Output

Each algorithm’s results are compared to the baseline `MPI_Alltoall`.

Example output:

```
Rank 0 sent: 0 1 2 3
Rank 1 sent: 100 101 102 103
Rank 2 sent: 200 201 202 203
Rank 3 sent: 300 301 302 303

[Pairwise XOR Exchange] SUCCESS: Results match the MPI_Alltoall reference.
[Linear Exchange] SUCCESS: All results are correct.
[Torsten Exchange] SUCCESS: Matches the MPI_Alltoall reference.
```

If the XOR method is skipped (when process count ≠ power of 2), the code will print:

```
XOR exchange cannot be done because: number of processes : 6 is not a power of 2.
```

---

## Build and Run

### Compile

```bash
mpic++ alltoall_algorithms.cpp -o alltoall
```

### Run (example with 4 processes)

```bash
mpirun -np 4 ./alltoall
```

---

## Performance Notes

| Algorithm                | Complexity | Power-of-2 Required | Non-blocking | Typical Use                       |
| ------------------------ | ---------- | ------------------- | ------------ | --------------------------------- |
| XOR Exchange             | O(log P)   | ✅ Yes              | ❌           | Fast when P is power of 2         |
| Linear Exchange          | O(P)       | ❌ No               | ❌           | Simple, general-purpose           |
| Torsten Hoefler Exchange | O(P)       | ❌ No               | ✅           | Efficient for large-scale systems |

---

# Assignment 2: MPI AllReduce Implementations: Linear, Ring, and Rabenseifner

This program implements and compares three custom `AllReduce` algorithms—**linear exchange**, **ring**, and **Rabenseifner**—against the built-in `MPI_Allreduce`. Each process starts with its own data segment, participates in reductions, and verifies correctness by comparing results to the MPI reference.

---

## Algorithms

### Linear Exchange AllReduce

- Two-phase method: **Reduce-Scatter** followed by **AllGather**.
- Uses `MPI_Isend` and `MPI_Irecv` with `MPI_Waitall` for synchronization.
- Works for any process count.
- Complexity: O(P).

### Ring AllReduce

- Data circulates around processes twice: first to reduce, then to broadcast results.
- Each step sends one chunk to the left and receives from the right.
- High bandwidth efficiency due to pipelining.
- Complexity: O(P).

### Rabenseifner AllReduce

- Combines **recursive halving** (reduce) and **recursive doubling** (broadcast).
- Requires process count to be a power of two.
- Complexity: O(log P).

---

## Helper Functions

- `is_power_of_two`: Checks power-of-two process counts.
- `cmp_buffers`: Verifies buffer equality.
- `print_buffer`: Prints rank-wise results.
- `print_function_docs`: Describes purpose and MPI usage of all functions.

---

## Build and Run

Compile:

```bash
mpic++ allreduce_algorithms.cpp -o allreduce
```

Run with 4 processes:

```bash
mpirun -np 4 ./allreduce
```

---

## Example Output

```
rank 0 linear_exchange passed
rank 1 linear_exchange passed
rank 2 ring passed
rank 3 rabenseifner passed
```

---

## Complexity Summary

| Algorithm       | Complexity | Power-of-2 Required | Non-blocking |
| --------------- | ---------- | ------------------- | ------------ |
| Linear Exchange | O(P)       | No                  | Yes          |
| Ring            | O(P)       | No                  | Yes          |
| Rabenseifner    | O(log P)   | Yes                 | Yes          |

---
