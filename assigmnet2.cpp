#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <cstring>
#define CHUNK_SIZE 2

struct FunctionDoc
{
    const char *name;
    const char *action;
    const char *theory;
    const char *mpi_usage;
};

void print_function_docs(int rank)
{
    if (rank != 0)
        return;
    static const FunctionDoc docs[] = {
        {"is_power_of_two", "Detects whether an integer is a power of two via bit masking.", "Binary powers of two have a single high bit, so n & (n-1) clears it only when n is a power of two.", "No MPI; helper for control flow."},
        {"linear_exchange_allReduce", "Runs a linear-phase reduce-scatter followed by allgather across ranks.", "Allreduce is decomposed into chunked reductions then broadcasts, matching linear exchange theory.", "Uses MPI_Isend/MPI_Irecv pairs plus MPI_Waitall in both phases."},
        {"ring_allReduce", "Implements a pipelined ring to accumulate and distribute chunk sums.", "Ring theory shifts chunks around a logical cycle to sum then circulate final results.", "Employs MPI_Isend to predecessors and MPI_Recv from successors per step."},
        {"rabenseifner_allReduce", "Executes Rabenseifner's logarithmic allreduce via recursive halving and doubling.", "Recursive halving reduces partners in log2(p) stages, then recursive doubling broadcasts the totals.", "Uses MPI_Isend/MPI_Recv with log2(p) peer exchanges each direction."},
        {"cmp_buffers", "Checks if two result buffers match element-wise.", "Straight comparison validates correctness of collectives.", "No MPI primitives involved."},
        {"print_buffer", "Outputs a labelled buffer for visual inspection.", "Pure I/O convenience for observing final vectors.", "No MPI involvement."},
        {"main", "Sets up MPI, seeds data, runs reference and custom allreduces, and reports.", "Demonstrates the collective algorithms against MPI_Allreduce for verification.", "Calls MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Allreduce, MPI_Barrier, and MPI_Finalize."}};
    for (const auto &doc : docs)
    {
        std::cout << doc.name << ":\n"
                  << "  action: " << doc.action << "\n"
                  << "  theory: " << doc.theory << "\n"
                  << "  mpi: " << doc.mpi_usage << "\n";
    }
    std::cout.flush();
}

bool is_power_of_two(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

int linear_exchange_allReduce(int rank, int n_procs, int n_elements, const std::vector<int> &sendbuf, std::vector<int> &recvbuf)
{
    int chunk_size = n_elements / n_procs;
    int reduce_scatter_tag = 0;
    std::vector<MPI_Request> send_req1(n_procs);
    std::vector<MPI_Request> recv_req1(n_procs);

    for (int phase = 0; phase < n_procs; phase++)
    {
        int send_peer = (rank + phase) % n_procs;
        int recv_peer = (rank - phase + n_procs) % n_procs;
        MPI_Isend(&sendbuf[send_peer * chunk_size], chunk_size, MPI_INT, send_peer, reduce_scatter_tag, MPI_COMM_WORLD, &send_req1[send_peer]);
        MPI_Irecv(&recvbuf[recv_peer * chunk_size], chunk_size, MPI_INT, recv_peer, reduce_scatter_tag, MPI_COMM_WORLD, &recv_req1[recv_peer]);
    }
    MPI_Waitall(n_procs, recv_req1.data(), MPI_STATUSES_IGNORE);

    std::vector<int> reduced_sum(chunk_size, 0);
    for (int idx = 0; idx < chunk_size; idx++)
    {
        for (int i = idx; i < n_elements; i += chunk_size)
            reduced_sum[idx] += recvbuf[i];
    }

    int all_gather_tag = 1;
    std::vector<MPI_Request> send_req2(n_procs);
    std::vector<MPI_Request> recv_req2(n_procs);
    for (int phase = 0; phase < n_procs; phase++)
    {
        int send_peer = (rank + phase) % n_procs;
        int recv_peer = (rank - phase + n_procs) % n_procs;
        MPI_Isend(reduced_sum.data(), chunk_size, MPI_INT, send_peer, all_gather_tag, MPI_COMM_WORLD, &send_req2[phase]);
        MPI_Irecv(&recvbuf[recv_peer * chunk_size], chunk_size, MPI_INT, recv_peer, all_gather_tag, MPI_COMM_WORLD, &recv_req2[phase]);
    }

    MPI_Waitall(n_procs, recv_req2.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(n_procs, send_req1.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(n_procs, send_req2.data(), MPI_STATUSES_IGNORE);
    return 0;
}

int ring_allReduce(int rank, int n_procs, int n_elements, const std::vector<int> &sendbuf, std::vector<int> &recvbuf)
{
    int chunk_size = n_elements / n_procs;
    int send_peer = (rank - 1 + n_procs) % n_procs;
    int recv_peer = (rank + 1) % n_procs;
    recvbuf = sendbuf; // copy initial data

    std::vector<MPI_Request> send_req((n_procs - 1) * 2);

    for (int step = 0; step < (n_procs - 1) * 2; step++)
    {
        int recv_chunk = (-rank - step - 1 + (3 * n_procs)) % n_procs;
        int send_chunk = (-rank - step + (3 * n_procs)) % n_procs;
        std::vector<int> recvbuf_step(chunk_size);

        MPI_Isend(&recvbuf[send_chunk * chunk_size], chunk_size, MPI_INT, send_peer, step, MPI_COMM_WORLD, &send_req[step]);
        MPI_Recv(recvbuf_step.data(), chunk_size, MPI_INT, recv_peer, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (step < n_procs - 1)
        {
            for (int i = 0; i < chunk_size; i++)
                recvbuf[recv_chunk * chunk_size + i] += recvbuf_step[i];
        }
        else
        {
            for (int i = 0; i < chunk_size; i++)
                recvbuf[recv_chunk * chunk_size + i] = recvbuf_step[i];
        }
    }

    MPI_Waitall((n_procs - 1) * 2, send_req.data(), MPI_STATUSES_IGNORE);
    return 0;
}

int rabenseifner_allReduce(int rank, int n_procs, int n_elements, const std::vector<int> &sendbuf, std::vector<int> &recvbuf)
{
    recvbuf = sendbuf;
    int send_chunk = 0, recv_chunk = 0, step;
    int n_steps = (int)log2(n_procs);
    std::vector<MPI_Request> send_req1(n_steps);

    for (step = 0; step < n_steps; step++)
    {
        int this_idx = (rank / (int)pow(2, step)) % 2;
        int peer = rank + (int)pow(2, step) * (this_idx == 1 ? -1 : 1);
        int chunk_size = n_elements / (int)pow(2, step + 1);
        recv_chunk += chunk_size * this_idx;
        send_chunk = recv_chunk + (chunk_size * (this_idx == 1 ? -1 : 1));
        std::vector<int> recvbuf_step(chunk_size);

        MPI_Isend(&recvbuf[send_chunk], chunk_size, MPI_INT, peer, step, MPI_COMM_WORLD, &send_req1[step]);
        MPI_Recv(recvbuf_step.data(), chunk_size, MPI_INT, peer, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < chunk_size; i++)
            recvbuf[recv_chunk + i] += recvbuf_step[i];
    }

    send_chunk = recv_chunk;
    step -= 1;
    std::vector<MPI_Request> send_req2(n_steps);

    for (; step >= 0; step--)
    {
        int this_idx = (rank / (int)pow(2, step)) % 2;
        int peer = rank + (int)pow(2, step) * (this_idx == 1 ? -1 : 1);
        int chunk_size = n_elements / (int)pow(2, step + 1);
        recv_chunk = send_chunk + (chunk_size * (this_idx == 1 ? -1 : 1));
        std::vector<int> recvbuf_step(chunk_size);

        MPI_Isend(&recvbuf[send_chunk], chunk_size, MPI_INT, peer, step + n_steps, MPI_COMM_WORLD, &send_req2[step]);
        MPI_Recv(recvbuf_step.data(), chunk_size, MPI_INT, peer, step + n_steps, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < chunk_size; i++)
            recvbuf[recv_chunk + i] = recvbuf_step[i];

        send_chunk -= chunk_size * this_idx;
    }

    MPI_Waitall(n_steps, send_req1.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(n_steps, send_req2.data(), MPI_STATUSES_IGNORE);
    return 0;
}

bool cmp_buffers(const std::vector<int> &buf1, const std::vector<int> &buf2)
{
    return buf1 == buf2;
}

void print_buffer(int rank, const std::vector<int> &buf, const std::string &algo_name)
{
    std::cout << "rank " << rank << " " << algo_name << " final buffer: ";
    for (auto val : buf)
        std::cout << val << " ";
    std::cout << "\n";
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    print_function_docs(rank);

    int n_ele = n_procs * CHUNK_SIZE;
    std::vector<int> sendbuf(n_ele, 0);
    std::vector<int> recvbuf(n_ele, 0);

    // Initialize distributed data
    for (int idx = 0; idx < CHUNK_SIZE; idx++)
        sendbuf[rank * CHUNK_SIZE + idx] = rank * 10 + idx + 1;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), n_ele, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // linear_exchange
    std::vector<int> recvbuf_linear_exchange(n_ele, 0);
    linear_exchange_allReduce(rank, n_procs, n_ele, sendbuf, recvbuf_linear_exchange);
    if (cmp_buffers(recvbuf_linear_exchange, recvbuf))
    {
        std::cout << "rank " << rank << " linear_exchange passed\n";
        print_buffer(rank, recvbuf_linear_exchange, "linear_exchange");
    }
    else
        std::cout << "rank " << rank << " linear_exchange FAILED\n";

    std::vector<int> recvbuf_ring_allReduce(n_ele, 0);
    ring_allReduce(rank, n_procs, n_ele, sendbuf, recvbuf_ring_allReduce);
    if (cmp_buffers(recvbuf_ring_allReduce, recvbuf))
    {
        std::cout << "rank " << rank << " ring passed\n";
        print_buffer(rank, recvbuf_ring_allReduce, "ring");
    }
    else
        std::cout << "rank " << rank << " ring FAILED\n";

    std::vector<int> recvbuf_rabenseifner(n_ele, 0);
    if (is_power_of_two(n_procs))
    {
        rabenseifner_allReduce(rank, n_procs, n_ele, sendbuf, recvbuf_rabenseifner);
        if (cmp_buffers(recvbuf_rabenseifner, recvbuf))
        {
            std::cout << "rank " << rank << " rabenseifner passed\n";
            print_buffer(rank, recvbuf_rabenseifner, "rabenseifner");
        }
        else
            std::cout << "rank " << rank << " rabenseifner FAILED\n";
    }
    else
    {
        std::cout << "rank " << rank << " rabenseifner skipped (not power-of-2 procs)\n";
    }

    MPI_Finalize();
    return 0;
}
