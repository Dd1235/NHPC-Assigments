#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// XOR pairwise exchange algorithm
bool xor_exchange(int rank, int size, vector<int> &recvbuf)
{
    if ((size & (size - 1)) != 0)
    {
        if (rank == 0)
        {
            cout << "XOR exchange cannot be done because: number of processes :" << size
                 << " is not a power of 2." << endl;
            cout << "Rank ^ phase may produce invalid partners." << endl;
        }
        return false; // xor not done
    }

    for (int phase = 1; phase < size; phase++)
    {
        int partner = rank ^ phase;

        int sendval = recvbuf[partner];
        int recvval = -1;

        MPI_Sendrecv(&sendval, 1, MPI_INT, partner, 0,
                     &recvval, 1, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        recvbuf[partner] = recvval;

        MPI_Barrier(MPI_COMM_WORLD);
    }
    return true;
}

// linear all to all exchange

void linear_exchange(int rank, int size, vector<int> &recvbuf)
{
    vector<int> tmpbuf = recvbuf;
    for (int step = 1; step < size; step++)
    {
        int send_to = (rank + step) % size;
        int recv_from = (rank - step + size) % size;

        int sendval = recvbuf[send_to];
        int recvval = -1;

        MPI_Sendrecv(&sendval, 1, MPI_INT, send_to, 0,
                     &recvval, 1, MPI_INT, recv_from, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        tmpbuf[recv_from] = recvval;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    recvbuf = tmpbuf;
}

// Torsten Hoefler exchange algorithm
void torsten_exchange(int rank, int size, int max_size,
                      int subtree_count,
                      const vector<int> &sendbuf,
                      vector<int> &recvbuf)
{
    int subtree_size = size / subtree_count;
    vector<MPI_Request> recv_request(size);
    vector<MPI_Request> send_request(size);

    for (int phase = 0; phase < size; phase++)
    {
        int subtree = ((rank % subtree_count) + (phase % subtree_count)) % subtree_count;
        int subtree_selection = ((rank / subtree_count) + (phase / subtree_count)) % subtree_size;
        int send_peer = (subtree * subtree_size) + subtree_selection;

        int recv_peer =
            (((rank % subtree_size) - (phase / subtree_count) + subtree_size) % subtree_size) * subtree_count +
            (((rank / subtree_size) - (phase % subtree_count) + subtree_count) % subtree_count);

        MPI_Irecv(&recvbuf[recv_peer], max_size / size, MPI_INT,
                  recv_peer, 0, MPI_COMM_WORLD, &recv_request[phase]);
        MPI_Isend(&sendbuf[send_peer], max_size / size, MPI_INT,
                  send_peer, 0, MPI_COMM_WORLD, &send_request[phase]);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    vector<MPI_Status> recv_status(size);
    vector<MPI_Status> send_status(size);
    MPI_Waitall(size, recv_request.data(), recv_status.data());
    MPI_Waitall(size, send_request.data(), send_status.data());
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sendbuf(size);
    for (int i = 0; i < size; i++)
        sendbuf[i] = rank * 100 + i;

    // Print send buffer
    vector<int> recvbuf = sendbuf;
    cout << "Rank " << rank << " sent: ";
    for (auto val : sendbuf)
        cout << val << " ";
    cout << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Default MPI_Alltoall
    MPI_Alltoall(sendbuf.data(), 1, MPI_INT,
                 recvbuf.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    cout << "Rank " << rank << " received (MPI_Alltoall): ";
    for (auto val : recvbuf)
        cout << val << " ";
    cout << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // xor exchange
    recvbuf = sendbuf;
    bool xor_done = xor_exchange(rank, size, recvbuf);

    // Check correctness for XOR
    bool xor_correct = true;
    if (xor_done)
    {
        vector<int> reference(size);
        for (int i = 0; i < size; i++)
            reference[i] = i * 100 + rank;
        for (int i = 0; i < size; i++)
        {
            if (recvbuf[i] != reference[i])
            {
                xor_correct = false;
                break;
            }
        }
    }

    // linear exchange
    recvbuf = sendbuf;
    linear_exchange(rank, size, recvbuf);

    bool linear_correct = true;
    vector<int> reference(size);
    for (int i = 0; i < size; i++)
        reference[i] = i * 100 + rank;
    for (int i = 0; i < size; i++)
    {
        if (recvbuf[i] != reference[i])
        {
            linear_correct = false;
            break;
        }
    }

    // torsten exchange
    recvbuf = sendbuf;
    torsten_exchange(rank, size, size, 2, sendbuf, recvbuf);

    bool torsten_correct = true;
    for (int i = 0; i < size; i++)
    {
        if (recvbuf[i] != reference[i])
        {
            torsten_correct = false;
            break;
        }
    }

    if (rank == 0)
    {
        // Report the outcome of the XOR (pairwise) exchange
        if (xor_done)
        {
            cout << "[Pairwise XOR Exchange] "
                 << (xor_correct ? "SUCCESS: Results match the MPI_Alltoall reference."
                                 : "FAILURE: Mismatches found compared to MPI_Alltoall.")
                 << endl;
        }
        else
        {
            cout << "[Pairwise XOR Exchange] Skipped (requires power-of-two number of processes)."
                 << endl;
        }

        // Report the outcome of the linear exchange
        cout << "[Linear Exchange] "
             << (linear_correct ? "SUCCESS: All results are correct."
                                : "FAILURE: Incorrect results detected.")
             << endl;

        // Report the outcome of the Torsten bandwidth-optimal exchange
        cout << "[Torsten Exchange] "
             << (torsten_correct ? "SUCCESS: Matches the MPI_Alltoall reference."
                                 : "FAILURE: Mismatches detected.")
             << endl;
    }

    MPI_Finalize();
    return 0;
}
