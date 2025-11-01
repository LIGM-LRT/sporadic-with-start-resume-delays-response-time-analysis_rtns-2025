#include <cassert>
#include <sys/sysinfo.h>
#include <sys/types.h>
import std;

#define KRST "\e[0m"
#define KBLD "\e[1m"
#define KDIM "\e[2m"
#define KREG "\e[22m"
#define KRED "\e[31m"
#define KGRN "\e[32m"
#define KYLW "\e[33m"
#define KBLU "\e[34m"
#define KMAJ "\e[35m"
#define KCYN "\e[36m"

using size_t = std::size_t;
using u64 = std::uint64_t;
using period_t = std::uint16_t;
using compute_t = std::uint16_t;
using util_t = std::uint64_t;
using response_time_t = period_t;
using period_vec = std::vector<period_t>;
using compute_vec = std::vector<compute_t>;
using generated_instance_v1 = std::tuple<u64, compute_vec, period_vec>;

class Utils {
public:
    static std::string print_bool(bool b, bool human = true) {
        std::string s;
        if (human) {
            s = (b ? "✅": "❌");
        } else {
            s = (b ? 'T' : 'F');
        }
        return s;
    }

    template <typename T, std::size_t N>
    static T get_val_or_last(const std::array<T, N> &arr, size_t idx){
        return ((idx < arr.size()) ? arr[idx] : arr.back());
    }
};

class Taskset {
public:
    const compute_vec c;
    const period_vec p;
    const compute_vec sd;
    const period_vec rd;
    const size_t sz;
    const period_vec ld_response_time;

    size_t size() const {
        return sz;
    }

    bool is_ld_schedulable() {
        for (size_t i=0; i<sz; i++) {
            if (ld_response_time[i] > p[i]) {
                return false;
            }
        }
        return true;
    }

    static inline u64 ceildiv(u64 p, u64 q) {
        return (p + q - 1) / q;
    }

    period_t compute_rild(size_t i, bool with_delays = true, bool unlocked = false) const {
        /* Note: even without delays, we reintegrate starting-delays in compute */
        u64 base_time = sd[i] + c[i];
        u64 rild = base_time;
        u64 limit = p[i];
        if (unlocked) {
            u64 unlocked_limit = 0x1000; //TODO: really arbitrary...
            limit = std::max(limit, unlocked_limit);
        }

        while (rild <= limit) {
            u64 s = 0;
            period_t max_ld = 0;
            if (with_delays) {
                max_ld = std::max(sd[i], rd[i]);
            }
            for (ssize_t j=i-1; j>=0; j--) {
                compute_t xd = sd[i];
                if ((sd[i] == 0) && (rd[i] > 0) && (c[i] > 1)) {
                    xd = 1; //TODO: prove this
                }
                if (!with_delays) {
                    xd = 0;
                }
                #ifdef OVERRIDE_OLD_FORMULA
                    xd = 0;
                #endif
                u64 nb_prempts = ceildiv(rild - xd, p[j]);
                s += nb_prempts * ((u64)sd[j] + c[j] + max_ld);
                if (with_delays) {
                    max_ld = std::max(max_ld, sd[j]);
                    max_ld = std::max(max_ld, rd[j]);
                }
            }
            u64 new_rild = base_time + s;
            if (new_rild == rild) {
                break;
            }
            rild = new_rild;
        }
        if (rild > p[i]) {
            rild = p[i] + 1;
        }
        return rild;
    }

    double compute_badldutil() const {
        u64 shoot = 1'000'000;

        u64 outcome = 0;
        period_t max_ld = 0;

        for (ssize_t j=sz-1; j>=0; j--) {
            u64 nb_prempts = ceildiv(shoot, p[j]);
            outcome += nb_prempts * ((u64)sd[j] + c[j] + max_ld);
            max_ld = std::max(max_ld, sd[j]);
            max_ld = std::max(max_ld, rd[j]);
        }

        return (double)outcome/(double)shoot;
    }

    double compute_ldutil() const {
        double ldutil = 0.;
        for(size_t i=0; i<sz; i++) {
            ldutil += (double)((u64)sd[i]+rd[i]) / (double)p[i];
        }
        return ldutil;
    }

    period_vec compute_ld_response_time() {
        period_vec ldrt;
        ldrt.reserve(sz);
        for (size_t i=0; i<sz; i++) {
            ldrt.push_back(compute_rild(i));
        }
        return ldrt;
    }

    inline compute_vec make_compute(const compute_vec &_c, const period_vec &_sd, bool deduce_sd) {
        if (!deduce_sd) {
            return _c;
        }
        auto new_c = _c;
        for (size_t i=0; i<new_c.size(); i++) {
            if (_sd[i] >= new_c[i]) {
                std::println(std::cerr, "Start delay larger or equal than compute in 'deduce mode' is not allowed");
                std::abort();
            }
            new_c[i] -= _sd[i];
        }
        return new_c;
    }

    Taskset(const compute_vec &_c, const period_vec &_p,
            const period_vec &_sd, const period_vec &_rd, bool deduce_sd)
        : c(make_compute(_c, _sd, deduce_sd)), p(_p), sd(_sd), rd(_rd),
          sz(c.size()), ld_response_time(compute_ld_response_time()) {}
};

template <>
struct std::formatter<Taskset> : std::formatter<string_view> {
    auto format(const Taskset& ts, std::format_context& ctx) const {
        std::string tmp;
        std::format_to(std::back_inserter(tmp), "[");

        for (size_t i=0; i<ts.size(); i++) {
            if (i > 0) {
                std::format_to(std::back_inserter(tmp), ", ");
            }
            std::format_to(std::back_inserter(tmp), "(C: " KCYN "{:2}" KRST ", P: " KCYN KBLD "{:2}" KRST,
                ts.c[i], ts.p[i], ts.sd[i], ts.rd[i]);
            if (i > 0) {
                bool had_sd = (ts.sd[i] > 0);
                bool had_rd = (ts.rd[i] > 0);
                std::format_to(std::back_inserter(tmp), ", {}SD: {:2}{}", (had_sd ? "" : KDIM), ts.sd[i], (had_sd ? "" : KREG));
                std::format_to(std::back_inserter(tmp), ", {}RD: {:2}{}", (had_rd ? "" : KDIM), ts.rd[i], (had_rd ? "" : KREG));
                std::format_to(std::back_inserter(tmp), ", " KDIM "rld: ");
                if (ts.ld_response_time[i] > ts.p[i]) {
                    std::format_to(std::back_inserter(tmp), KRED "XX");
                } else {
                    std::format_to(std::back_inserter(tmp), "{:2}", ts.ld_response_time[i]);
                }
            }
            std::format_to(std::back_inserter(tmp), KRST ")");
        }

        std::format_to(std::back_inserter(tmp), "]");

        return std::formatter<string_view>::format(tmp, ctx);
    }
};

class GeneratedInstanceV2 {
public:
    u64 id;
    compute_vec c;
    period_vec p;
    compute_vec sd;
    compute_vec rd;
    static constexpr u64 END_WEIGHT = std::numeric_limits<u64>::max();
    u64 weight;
    bool pn_frozen = false;
    bool cn_frozen = false;
    bool sdn_frozen = false;

    [[gnu::pure]] inline
    u64 partial_weight(size_t index, period_t period) {
        bool USE_COST_ONE = true;
        if (USE_COST_ONE) {
            return period;
        }
        return index * period;
    }

    u64 compute_weight() {
        /* weight: sum(cost(i) * period_i), with cost(i)=i or 1 */
        u64 w = 0;
        size_t idx = 1;
        for (const auto &pi: p) {
            w += partial_weight(idx, pi);
            idx++;
        }
        return w;
    }

    inline
    void add_weight_period_increment() {
        size_t idx = p.size();
        weight += partial_weight(idx, p[idx - 1]) - partial_weight(idx, p[idx - 1] - 1);
    }

    inline
    void add_weight_new_task() {
        size_t idx = p.size();
        weight += partial_weight(idx, p[idx - 1]);
    }

    // TODO: remember the period_lcm of the first tasks which are frozen.
    static util_t period_lcm(const period_vec &p) {
        util_t lcm = p[0];
        for (size_t i=1; i<p.size(); i++) {
            util_t pi = p[i];
            static_assert(std::is_same<decltype(lcm), unsigned long int>::value);
            util_t gcd_i = std::gcd(lcm, pi);
            bool overflow = __builtin_umull_overflow(lcm, pi/gcd_i, &lcm);
            if (overflow) {
                std::println(std::cerr, "Overflow in {}", __func__);
                std::abort();
            }
        }
        return lcm;
    }

    std::pair<util_t, util_t> compute_utilization() {
        util_t utilization_one = period_lcm(p); /* Avoid floats for utilization */
        util_t utilization = 0;
        for (size_t i=0; i<p.size(); i++) { /* sum c_i/p_i */
            static_assert(std::is_same<decltype(utilization), unsigned long int>::value);
            util_t to_add;
            bool overflow = __builtin_umull_overflow(c[i], utilization_one, &to_add);
            to_add /= p[i];
            overflow |= __builtin_uaddl_overflow(utilization, to_add, &utilization);
            if (overflow) {
                std::println(std::cerr, "Overflow in {}", __func__);
                std::abort();
            }
        }

        return std::make_pair(utilization, utilization_one);
    }

    GeneratedInstanceV2(u64 _id, const compute_vec &_c, const period_vec &_p,
                        const compute_vec &_sd, const compute_vec &_rd)
     : id(_id), c(_c), p(_p), sd(_sd), rd(_rd), weight(compute_weight()) {}
};

template <typename T>
class WaitingQueue {
    using elem_t = T;
    std::deque<elem_t> queue;
protected:
    static constexpr std::chrono::duration sleep_thread_for_more_work = std::chrono::milliseconds(100);
    static constexpr size_t waiting_queue_max_size = 0x1000;
    std::mutex mtx;

public:
    void push(const elem_t &elem) {
        mtx.lock();
        while (queue.size() >= waiting_queue_max_size) {
            mtx.unlock();
            std::this_thread::sleep_for(sleep_thread_for_more_work);
            mtx.lock();
        }
        queue.push_back(elem);
        mtx.unlock();
    }

    elem_t pop() {
        mtx.lock();

        while (queue.empty()) {
            mtx.unlock();
            std::this_thread::sleep_for(sleep_thread_for_more_work);
            mtx.lock();
        }

        elem_t elem = queue.front();
        queue.pop_front();

        mtx.unlock();

        return elem;
    }
};
using WaitingQueueV1 = WaitingQueue<generated_instance_v1>;

class WeightedWaitingQueue : WaitingQueue<GeneratedInstanceV2> {
    using elem_v2_t = GeneratedInstanceV2;
    std::priority_queue<elem_v2_t, std::vector<elem_v2_t>,
                        decltype([](elem_v2_t& a, elem_v2_t& b){return a.weight > b.weight;})> queue;
    size_t max_weight;

public:
    WeightedWaitingQueue(size_t _max_weight, size_t nb_threads) : max_weight(_max_weight) {
        auto end_elem = elem_v2_t(0, {}, {}, {}, {});
        end_elem.weight = end_elem.END_WEIGHT;
        for (size_t i=0; i<nb_threads; i++) {
            push_no_checkweight(end_elem);
        }
    }

    inline
    void push_no_checkweight(const elem_v2_t &elem) {
        mtx.lock();
        queue.push(elem);
        mtx.unlock();
    }

    void push(const elem_v2_t &elem) {
        if (elem.weight > max_weight) {
            return;
        }
        push_no_checkweight(elem);
    }

    elem_v2_t pop() {
        mtx.lock();

        while (queue.empty()) {
            mtx.unlock();
            std::this_thread::sleep_for(sleep_thread_for_more_work);
            mtx.lock();
        }

        elem_v2_t elem = queue.top();
        queue.pop();

        mtx.unlock();

        return elem;
    }

    bool is_empty() {
        mtx.lock();
        bool ret = queue.empty();
        mtx.unlock();
        return ret;
    }
};


class SimulationSporadic {
public:
    static constexpr size_t MAX_TASKS =
    #ifdef OVERRIDE_MAX_TASKS
        OVERRIDE_MAX_TASKS
    #else
        8
    #endif
    ;
private:
    using period_arr = std::array<period_t, MAX_TASKS>;
    using compute_arr = std::array<compute_t, MAX_TASKS>;
    using sdd_t = std::uint16_t; static_assert(MAX_TASKS <= 16);

    class State {
    public:
        /*
         * State of system:
         * - forall task i:
         *   - how long ago the job started (range [0; Ti+1]),
         *     Note: Ti+1 is only used in periodic+all-possible-offsets mode.
         *           It means the task has never started.
         *   - if starting delay has been performed,
         *   - compute left for job (range [0; Ci]).
         * - how much LD is left for the running task
         */

        period_arr jobs_started;
        sdd_t start_delays_done;
        compute_arr computes_left;
        compute_t ld_left;
#ifdef BF
        double bf_distance; /* Bellman-ford internal state, must not be hashed */
#endif /* BF */

    public:
        inline
        bool get_sdd(size_t i) const {
            return (start_delays_done >> i) & 0b1u;
        }

        inline
        void set_sdd(size_t i, bool b) {
            if (b) {
                start_delays_done |= (1u << i);
            } else {
                start_delays_done &= ~(1u << i);
            }
        }

        inline
        double get_bf_distance() const {
            #ifdef BF
            return bf_distance;
            #endif
            return 0.;
        }

        inline
        void set_bf_distance([[gnu::unused]] double d) const {
            #ifdef BF
            bf_distance = d;
            #endif
        }

        std::string to_str(size_t n) const {
            std::ostringstream ss;
            ss << "State(JS:" KCYN "(";
            bool first = true;
            for (size_t i=0; i<n; i++) {
                if (first) {
                    first = false;
                } else {
                    ss << ",";
                }
                std::print(ss, "{:2}", jobs_started[i]);
            }
            ss << ")" KRST ",SDD(";
            first = true;
            for (size_t i=0; i<n; i++) {
                bool sdd = get_sdd(i);
                if (first) {
                    first = false;
                } else {
                    ss << ",";
                }
                ss << KBLD <<(sdd ? 'T' : 'F') << KRST;
            }
            ss << "),";
            std::print(ss, "CL:" KBLD KCYN);
            first = true;
            for (size_t i=0; i<n; i++) {
                if (first) {
                    first = false;
                } else {
                    ss << ",";
                }
                std::print(ss, "{:2}", computes_left[i]);
            }
            std::print(ss, KRST ",ld_left:{})", ld_left);
            ss << ")";
            return ss.str();
        }

        bool operator==(const State& other) const {
            // TODO: not optimal, we compare unused bytes in "jobs_started" and "computes_left"
            return std::tie(jobs_started,start_delays_done,computes_left,ld_left) ==
                   std::tie(other.jobs_started,other.start_delays_done,other.computes_left,other.ld_left);
        }

        State(const period_arr &_jobs_started, const sdd_t &_start_delays_done,
              const compute_arr &_computes_left, compute_t _ld_left)
              : jobs_started(_jobs_started), start_delays_done(_start_delays_done),
                computes_left(_computes_left), ld_left(_ld_left) {
        }
    };

    struct HashState {
        template<typename T>
        static inline std::uint32_t rehash(std::uint32_t h, T v) {
            h = ((h >> 16) ^ h) * 0x45d9f3bu;
            h = ((h >> 16) ^ h) * 0x45d9f3bu;
            h = (h >> 16) ^ h;
            h ^= 0x9e3779b9u + v;
            return h;
        }

        std::size_t operator()(const State& state) const noexcept
        {
            std::uint32_t h = 0;
            // TODO: not optimal, we compare unused bytes in "jobs_started" and "computes_left"
            for (const auto& v: state.jobs_started) {
                h = rehash(h, v);
            }
            h = rehash(h, state.start_delays_done);
            for (const auto& v: state.computes_left) {
                h = rehash(h, v);
            }
            h = rehash(h, state.ld_left);
            return (size_t) h;
        }
    };

    const Taskset taskset;
    const size_t N;
    const bool only_periodic; /* Only simulate "periodic with offsets" scheduling */
    const bool only_full_compute; /* Don't allow a job to finish before its WCETS (compute & delays) */
    const int verbose;
    bool have_fixed_periodic_offset = false;
    period_vec periodic_offsets; /* set iff 'have_fixed_periodic_offset' */
    std::unordered_set<State, HashState> seen_states;
    std::unordered_set<State, HashState> new_states;
    std::vector<response_time_t> max_seen_response_time;
    bool found_shorter_path;

    inline
    void push_state(const State &state, bool construct_phase, double new_bf_distance) {
        bool seen = true;
        auto elem = seen_states.find(state);
        if (elem == seen_states.end()) {
            /* New element */
            seen = false;
            seen_states.insert(state);
            new_states.insert(state);
        } else if (!construct_phase) {
            if (new_bf_distance > std::abs(elem->get_bf_distance())) {
                /* 'bf_distance' is not hashed, so this is OK to remove the const */
                State &pelem = const_cast<State&>(*elem);
                pelem.set_bf_distance(-new_bf_distance); /* '<0.' means 'to handle again' */
                found_shorter_path = true;
            }
        }
        if (verbose > (0 + (construct_phase ? 0 : 1))) {
            std::println("    SS: push_state: {} {}  || seen:{} new:{}",
                (seen ? "SEEN" : "NEW "), state.to_str(N), seen_states.size(), new_states.size()
            );
        }
    }

    size_t get_running_task_index(compute_arr computes_left) {
        size_t i;
        for (i=0; i<computes_left.size(); i++) {
            if (computes_left[i] > 0) {
                break;
            }
        }
        return i; /* Might be equal to computes_left.size() if no running task */
    }

    period_arr build_initial_jobs_started() {
        period_arr initial_jobs_started;
        for (size_t i=0; i<N; i++) {
            /* Initialization: for sporadic, any task can start a job, none has any compute left */
            initial_jobs_started[i] = taskset.p[i];
            if (only_periodic) {
                if (have_fixed_periodic_offset) {
                    /* We only support an offset less or equal than the task period */
                    assert(periodic_offsets[i] <= initial_jobs_started[i]);
                    initial_jobs_started[i] -= periodic_offsets[i];
                } else {
                    initial_jobs_started[i]++;
                }
            }
        }
        return initial_jobs_started;
    }

    /*
     * From a given state (node), generate edges on the fly and:
     * - Create missing nodes
     * - Update Bellman-Ford state
     */
    inline
    bool traverse_simu_state(State &state, bool construct_phase, double lambda) {
        size_t running_task_index = get_running_task_index(state.computes_left);

        /* Try to start any job which could start */
        bool some_jobs_can_start = false;
        for (size_t task_index=0; task_index<N; task_index++) {
            if (state.jobs_started[task_index] < taskset.p[task_index]) {
                continue;
            }

            some_jobs_can_start = true;

            /* Cannot start a new job if the previous one did not finish */
            assert(state.computes_left[task_index] == 0);

            State new_state = state;
            /* Update job start */
            new_state.jobs_started[task_index] = 0;
            /* Reset to needing start delay */
            new_state.set_sdd(task_index, false);
            /* Reset compute for the new job */
            new_state.computes_left[task_index] = taskset.c[task_index];
            /* If the new task is the new running task, reset 'ld_left' */
            if (task_index < running_task_index) {
                new_state.ld_left = taskset.sd[task_index];
            }
            /* Edge distance is zero, time does not move here */
            push_state(new_state, construct_phase, state.get_bf_distance());
        }

        bool go_forward = true;
        if (only_periodic) {
            /*
                * We can only go forward if no job can start, except if the task itself
                *  has never started (periodic with offsets)
                */
            go_forward = !some_jobs_can_start;
            if (!go_forward) {
                /* For performance purposes, we might want a "all jobs started" flag in the state */
                for (size_t task_index=0; task_index<N; task_index++) {
                    if (state.jobs_started[task_index] > taskset.p[task_index]) {
                        go_forward = true;
                        break;
                    }
                }
            }
        }

        if (go_forward) {
            if (!only_full_compute && (running_task_index < N)) {
                /*
                    * Assume running task job terminates now.
                    * Actually, this is a bit more complex:
                    * - if there is delay left, early terminate the delay
                    * - otherwise, early terminate the compute
                    */

                auto saved_ld_left = state.ld_left;
                bool sdd = state.get_sdd(running_task_index);
                compute_t saved_compute_left;
                bool delay_early_finish;

                /*  Unlikely case where SD=0 and RD>0. See below */
                bool unlikely_rd_nosd = ((saved_ld_left == 0) && !state.get_sdd(running_task_index));
                if ((saved_ld_left > 0) || unlikely_rd_nosd) {
                    delay_early_finish = true;
                    state.ld_left = 0;
                    state.set_sdd(running_task_index, true);
                } else {
                    delay_early_finish = false;
                    saved_compute_left = state.computes_left[running_task_index];
                    state.computes_left[running_task_index] = 0;
                    state.set_sdd(running_task_index, false);
                }

                /* Edge distance is zero, time does not move here */
                push_state(state, construct_phase, state.get_bf_distance());

                if (delay_early_finish) {
                    state.ld_left = saved_ld_left;
                    if (!sdd) {
                        state.set_sdd(running_task_index, false);
                    }
                } else {
                    state.computes_left[running_task_index] = saved_compute_left;
                    if (sdd) {
                        state.set_sdd(running_task_index, true);
                    }
                }
            }

            /* Now handle the case where we actually go forward by one time-unit */
            double extra_bf_distance = 0.;
            if (running_task_index < N) {
                /* Update compute & delay status of the running task */
                if (state.ld_left == 0) {
                    if (!state.get_sdd(running_task_index)) {
                        /*
                         * Unlikely case where SD=0 and RD>0.
                         * Cannot be done before, since we must wait for the first 'real'
                         *  compute slot to register the zero-length starting delay.
                         */
                        state.set_sdd(running_task_index, true);
                    }

                    state.computes_left[running_task_index] -= 1;

                    if (state.computes_left[running_task_index] == 0) {
                        /*
                         * Job ended:
                         *  - update max response time
                         *  - reset sdd to false
                         *  - set ld_left for the next task
                         */
                        if (construct_phase) {
                            max_seen_response_time[running_task_index] = std::max(
                                max_seen_response_time[running_task_index],
                                (period_t)(state.jobs_started[running_task_index] + 1)
                            );
                        }

                        state.set_sdd(running_task_index, false);

                        assert(state.ld_left == 0);
                        size_t new_running_task_index = get_running_task_index(state.computes_left);
                        if (new_running_task_index < N) {
                            bool sdd = state.get_sdd(new_running_task_index);
                            if (sdd) {
                                state.ld_left = taskset.rd[new_running_task_index];
                            } else {
                                state.ld_left = taskset.sd[new_running_task_index];
                            }
                        }
                    }
                } else {
                    state.ld_left -= 1;
                    if (state.ld_left == 0 && !state.get_sdd(running_task_index)) {
                        state.set_sdd(running_task_index, true);
                    }
                    extra_bf_distance = 1.;
                }
            }

            /* Jobs started are now one unit older */
            for (size_t i=0; i<N; i++) {
                if (state.jobs_started[i] < taskset.p[i]) {
                    state.jobs_started[i]++;
                }
            }

            /* Check for deadline miss */
            if (construct_phase) {
                for (size_t i=0; i<N; i++) {
                    if ((state.computes_left[i] > 0) && (state.jobs_started[i] >= taskset.p[i])) {
                        if (verbose > 0) {
                            std::println("  SS: deadline miss for: {}", state.to_str(N));
                        }
                        return false;
                    }
                }
            }

            push_state(state, construct_phase, state.get_bf_distance() + extra_bf_distance - lambda);
        }
        return true;
    }

    /* Must be called *after* the graph has been constructed */
    bool has_ratio_of(double lambda) {
        /* Reset internal state */
        for (auto &state: seen_states) {
            /* 'bf_distance' is not hashed, so this is OK to remove the const */
            State &pstate = const_cast<State&>(state);
            pstate.set_bf_distance(0.0);
        }

        /* Check for a positive cycle with lambda */
        //TODO: optimize Bellman-Ford
        size_t nb_states = nb_simu_states();
        for (size_t i=0; i<nb_states; i++) {
            found_shorter_path = false;
            for (auto &state: seen_states) {
                if ((i>0) && (state.get_bf_distance() >= 0.)) {
                    continue; /* No change here, discard */
                }

                /* 'bf_distance' is not hashed, so this is OK to remove the const */
                State &pstate = const_cast<State&>(state);
                pstate.set_bf_distance(std::abs(pstate.get_bf_distance())); /* Mark it as handled */

                /* 'vstate' cannot be a reference here, as we modify it */
                auto vstate = pstate;
                traverse_simu_state(vstate, false, lambda);
            }
            assert (nb_states == nb_simu_states());
            if (!found_shorter_path) {
                break;
            }
        }

        return found_shorter_path;
    }

public:
    void set_offsets(const period_vec &offsets) {
        assert(only_periodic);
        assert(offsets.size() == N);
        have_fixed_periodic_offset = true;
        periodic_offsets = offsets;
    }

    size_t nb_simu_states() {
        return seen_states.size();
    }

    bool run() {
        if (verbose > 0) {
            std::println("  SS: run simulation");
        }

        seen_states.clear();
        new_states.clear();
        std::fill(max_seen_response_time.begin(), max_seen_response_time.end(), 0);

        compute_arr initial_computes_left{};
        sdd_t initial_start_delays_done{};
        const State initial_state(build_initial_jobs_started(),
                                  initial_start_delays_done, initial_computes_left, 0);
        if (verbose > 0) {
            std::println("  SS: pushing initial state");
        }
        push_state(initial_state, true, 0.);

        while (new_states.size() > 0) {
            auto it = new_states.begin();
            State state = *it;
            new_states.erase(it);
            if (verbose > 0) {
                std::println("  SS: POP {}", state.to_str(N));
            }

            bool ok = traverse_simu_state(state, true, 0.0);
            if (!ok) {
                return false; /* Deadline-miss spotted */
            }
        }

        return true;
    }

    /* Can only be called after a successful call to 'run' */
    double compute_ldratio() {
        constexpr int SKEW = 95;
#if 1 //TODO
        int lambda_min = 0, lambda_max = 100;
        while(lambda_min < lambda_max) {
            int lambda = lambda_min + SKEW * (lambda_max - lambda_min) / 100;
            if (lambda == lambda_min) {
                lambda++;
            }
            if (verbose > 0) {
                std::println("  SS: [{}; {}] --> lambda={}", lambda_min, lambda_max, lambda);
            }

            if (has_ratio_of(lambda/100.)) {
                lambda_min = lambda;
            } else {
                lambda_max = lambda - 1;
            }
        }

        return lambda_min/100.;
#else
        double lambda_min = 0., lambda_max = 1.;
        while(lambda_max - lambda_min > 0.01) {
            double lambda = lambda_min + SKEW/100. * (lambda_max - lambda_min);
            if (verbose > 0) {
                std::println("  SS: [{:.3f}; {:.3f}] --> lambda={:.3f}", lambda_min, lambda_max, lambda);
            }

            if (has_ratio_of(lambda)) {
                lambda_min = lambda;
            } else {
                lambda_max = lambda;
            }
        }

        return lambda_min;
#endif
    }

    const std::vector<response_time_t>& get_worst_response_times() const {
        return max_seen_response_time;
    }

    SimulationSporadic(const Taskset &_taskset, bool _only_periodic,
                       bool _only_full_compute, int _verbose)
        : taskset(_taskset), N(taskset.size()), only_periodic(_only_periodic),
          only_full_compute(_only_full_compute), verbose(_verbose) {
        assert(N <= MAX_TASKS);
        max_seen_response_time.resize(N);
    }
};

class Bruteforce {
    const unsigned nb_tasks;
    const compute_t min_delay;
    WaitingQueueV1 *queue;
    const int verbose;

    static util_t period_lcm(const period_vec &p) {
        util_t lcm = p[0];
        for (size_t i=1; i<p.size(); i++) {
            util_t pi = p[i];
            static_assert(std::is_same<decltype(lcm), unsigned long int>::value);
            util_t gcd_i = std::gcd(lcm, pi);
            bool overflow = __builtin_umull_overflow(lcm, pi/gcd_i, &lcm);
            if (overflow) {
                std::println(std::cerr, "Overflow in {}", __func__);
                std::abort();
            }
        }
        return lcm;
    }

    template<typename T>
    static T vec_gcd(std::vector<T> v) {
        util_t gcd = v[0];
        for (size_t i=1; i<v.size(); i++) {
            gcd = std::gcd(gcd, v[i]);
        }
        return gcd;
    }

public:
    Bruteforce(unsigned _nb_tasks, compute_t _min_delay, WaitingQueueV1 *_queue, int _verbose = 0)
        : nb_tasks(_nb_tasks), min_delay(_min_delay), queue(_queue), verbose(_verbose) {
    }

    /*
     * Generates tasksets by increasing sum of Ti.
     * Taskset that are knoww for sure to be not schedulabled are skipped.
     *'min_delay' is a lower bound on loading delays. Avoid generating tasksets with too low 'compute'
     */
    void generate() {
        /* Since a period of 0 or 1 is useless, start at 2 */
        constexpr period_t MIN_PERIOD = 2;

        u64 instance_idx = 0;
        for (u64 cost=0; /* no stop condition*/; cost++) {
            if (verbose > 0) {
                std::println("  CG: cost={}", cost);
            }
            period_vec p(nb_tasks, MIN_PERIOD); /* List of periods */
            p[0] += cost;

            ssize_t index = 0;
            if (cost == 0) {
                index = -1;
            }

            /* Generate all period lists such that sum(p) == cost */
            for (;;) {
                bool valid_period = true;
                for (size_t i=0; i<p.size(); i++) {
                    /*
                     * Simple pruning: jobs from tasks <i may run sequentially,
                     * and are at least `min_delay + 1` long, hence period of job i
                     * must be a least (i+1) * this long (aka <i plus itself).
                     * This should ideally be done during ripple. V2 does this.
                     */
                    if (p[i] < ((i + 1) * (min_delay + 1))) {
                        valid_period = false;
                        break;
                    }
                }
                /* We have periods, delegate compute generation. */
                if (valid_period) {
                    subgen_compute(instance_idx, p);
                }

                if (index < 0) {
                    break;
                }

                if (index + 1 < nb_tasks) {
                    /* ripple to next index */
                    p[index] -= 1;
                    index += 1;
                    p[index] += 1;
                    if (verbose > 0) {
                        std::println("    CG: ripple index={} p={}", index, p);
                    }
                } else {
                    /* move back p[-1] and ripple */
                    auto new_index = index - 1;
                    while (new_index >= 0 and p[new_index] == MIN_PERIOD) {
                        new_index -= 1;
                    }
                    if (new_index < 0) {
                        break;
                    }
                    period_t rightmost = p.back() - MIN_PERIOD;
                    p.back() = MIN_PERIOD;
                    p[new_index] -= 1;
                    new_index += 1;
                    p[new_index] += 1 + rightmost;
                    index = new_index;
                    if (verbose > 0) {
                        std::println("    CG: backt index={} p={}", index, p);
                    }
                }
            }
            assert(p.back() == MIN_PERIOD + cost);
        }
        if (verbose >= 0) {
            std::println("  CG: finished");
        }
    }

    /* Generate all computes for given periods */
    void subgen_compute(u64 &instance_idx, const period_vec &p) {
        compute_t min_compute = min_delay + 1;
        util_t utilization_one = period_lcm(p); /* Avoid floats for utilization */
        compute_vec c(p.size(), min_compute); /* Initialize compute to the minimum */
        util_t utilization = 0;
        for (size_t i=0; i<p.size(); i++) { /* sum c_i/p_i */
            static_assert(std::is_same<decltype(utilization), unsigned long int>::value);
            util_t to_add;
            bool overflow = __builtin_umull_overflow(c[i], utilization_one, &to_add);
            to_add /= p[i];
            overflow |= __builtin_uaddl_overflow(utilization, to_add, &utilization);
            if (overflow) {
                std::println(std::cerr, "Overflow in {}", __func__);
                std::abort();
            }
        }
        if (utilization > utilization_one) {
            return; /* Bail out if utilization > 1 */
        }
        if (verbose > 0) {
            std::println("      SG: p={}", p);
        }

        period_t p_gcd = vec_gcd(p);
        for (;;) {
            auto gcd = std::gcd(p_gcd,vec_gcd(c));
            if (verbose > 0) {
                std::println("        SG: c={} p={} gcd={}", c, p, gcd);
            }
            if (gcd > 1) {
                return;
            }

            generated_instance_v1 elem(instance_idx, c, p);
            queue->push(elem);
            instance_idx++;

            size_t index = 0;
            bool dirty = true;
            while (dirty && (index < c.size())) {
                dirty = false;
                utilization += utilization_one / p[index];
                c[index] += 1;
                if (c[index] >= p[index] or utilization > utilization_one) {
                    utilization -= (c[index] - min_compute) * utilization_one / p[index];
                    c[index] = min_compute;
                    index += 1;
                    dirty = true;
                }
                /*
                 * Ideally, we should baill out if either:
                 *  - instance not not regular schedulable
                 *  - previous simulation for an "easier" tasket failed.
                 * V2 does this.
                 */
            }
            if (index >= c.size()) {
                if (verbose > 0) {
                    std::println("      SG: --> OUT");
                }
                break;
            }
        }
    }
};

class TaskGenerator {
public:
    class GeneratedInstance {
    public:
        u64 id;
        compute_vec c;
        period_vec p;
        compute_vec sd;
        compute_vec rd;
        period_vec o; /* optional offsets in periodic case */
        double utilization;

        GeneratedInstance(u64 _id, size_t nb_tasks) {
            id = _id;
            c.resize(nb_tasks);
            p.resize(nb_tasks);
            sd.resize(nb_tasks);
            rd.resize(nb_tasks);
            o.resize(nb_tasks);
        }
    };

    enum PeriodGenerator {
        PeriodGeneratorDumb,
        PeriodGeneratorSmallHP,
    };

    enum SimulationType {
        SimuTypeSporadic = 0,
        SimuTypePeriodicAnyOffset = 1,
        SimuTypePeriodicGivenOffset = 2,
    };

    static constexpr bool CONFIG_SD_GEQ_RD = true;
    static constexpr bool CONFIG_ANY_SD = false;

    class Config {
    public:
        const u64 max_period;
        const SimulationType simu_type;
        const bool sd_geq_rd;
        const PeriodGenerator period_gen;

        Config(u64 _max_period, SimulationType _simu_type, bool _sd_geq_rd, PeriodGenerator _period_gen)
            : max_period(_max_period), simu_type(_simu_type), sd_geq_rd(_sd_geq_rd), period_gen(_period_gen) {}
    };

private:
    const unsigned nb_tasks;
    const Config config;
    const int verbose;
    std::mt19937 gen;
    u64 nb_gen = 0;

    u64 randunif(u64 min, u64 max) {
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }

    u64 randlog(u64 min, u64 max) {
        double logmin = 0.0;
        double logmax = std::log((double)(max-min+2));
        std::uniform_real_distribution<> dis(logmin, logmax);
        double ru = dis(gen);
        double rl = std::exp(ru);
        u64 r = ((u64) rl) - 1;
        if (verbose > 1) {
            std::println("  Randlog [{};{}[ -> [{};{}[ ru:{}, rl:{}, r:{}", min, max, logmin, logmax, ru, rl, r);
        }
        assert (rl >= 1.);
        assert (r <= max - min);
        return min + r;
    }

    u64 randfoldednorm(u64 min, u64 max) {
        double stddev = (double)(max - min) / 3.;
        std::normal_distribution dis{0., stddev};
        u64 r = max+1;
        while (r > max - min) {
            double rn = std::abs(dis(gen));
            r = (u64) rn;
        }
        assert (r <= max - min);
        return min + r;
    }

    u64 randchisquared(u64 min, u64 max) {
        std::chi_squared_distribution dis{1.};
        u64 r = max+1;
        while (r > max - min) {
            double rn = std::abs(dis(gen));
            r = (u64) rn;
        }
        assert (r <= max - min);
        return min + r;
    }

    u64 randmix(u64 min, u64 max, bool logchi) {
        if (logchi) {
            return randchisquared(min, max);
        }
        return randlog(min, max);
    }

    void check_randlog() {
        for (size_t x=0; x<10; x++) {
            for (size_t y=x; y<x+10; y++) {
                std::vector<u64> v;
                v.resize(y-x+1);
                for (size_t z=0; z<10'000; z++) {
                    auto r = randlog(x, y);
                    assert(r-x <= v.size());
                    v[r-x]++;
                }
                std::println("randlog({}, {}) ==> {}", x, y, v);
            }
        }
    }

    u64 gen_period_hpp() {
        constexpr std::array line_2  = {1, 2, 2, 2, 4, 4, 4, 8, 16, 16};
        constexpr std::array line_3  = {1, 3, 3, 3, 3, 9, 9, 27};
        constexpr std::array line_5  = {1, 5, 5, 5, 25};
        constexpr std::array line_7  = {1, 7, 7, 7, 7};
        constexpr std::array line_11 = {1, 11, 11, 11, 11};
        constexpr std::tuple matrix{line_2, line_3, line_5, line_7, line_11};

        u64 p = 1;
        std::apply([this, &p](auto&&... args)
            {
                ((p *= args[randunif(0, args.size()-1)]), ...);
            }, matrix
        );

        return p;
    }

    u64 gen_period() {
        switch(config.period_gen) {
            case PeriodGeneratorDumb:
                return randlog(nb_tasks, config.max_period);
            case PeriodGeneratorSmallHP:
                while (true) {
                    u64 p = gen_period_hpp();
                    if ((p > nb_tasks) && (p <= config.max_period)) {
                        return p;
                    }
                }
        }
        assert(false);
        __builtin_unreachable();
    }

public:
    TaskGenerator(unsigned _nb_tasks, u64 _seed, Config _config, int _verbose = 0)
        : nb_tasks(_nb_tasks), config(_config), verbose(_verbose) {
        gen.seed(_seed);
    }

    void generate(GeneratedInstance &elem) {
        bool do_check_randlog = false;
        if (do_check_randlog) {
            check_randlog();
            std::abort();
        }
        nb_gen++;
        bool logchi = ((nb_gen % 2) == 0);

        bool ok = false;
        while (!ok) {
            double utilization = 0;
            for (size_t i=0; i<nb_tasks; i++) {
                elem.p[i] = gen_period();
                elem.c[i] = randmix(1, elem.p[i], logchi);
                if (i == 0) {
                    elem.sd[i] = 0;
                    elem.rd[i] = 0;
                } else {
                    elem.sd[i] = randmix(0, elem.c[i], logchi);
                    elem.rd[i] = randmix(0, (config.sd_geq_rd ? elem.sd[i] : elem.c[i]), logchi);
                }
                elem.o[i] = randunif(0, elem.p[i]-1);

                utilization += ((double)elem.c[i] + (double)elem.sd[i]) / (double)elem.p[i];

                if (verbose > 0) {
                    std::println("MultiGen [{}] c:{} p:{} sd:{} rd:{} util:{:.2f}",
                                i, elem.c[i], elem.p[i], elem.sd[i], elem.rd[i], utilization);
                }

                if (utilization > 1.) {
                    break;
                }
            }

            if (utilization > 1.) {
                continue;
            }

            elem.utilization = utilization;
            ok = true;
        }
    }
};
using TG = TaskGenerator;

class FSimuV1 {
    static constexpr size_t END_ID = std::numeric_limits<size_t>::max();

    static void generator_thread(WaitingQueueV1 *queue, size_t nb_threads, unsigned nb_tasks, unsigned min_delay) {
        constexpr int VERBOSE = 0;

        Bruteforce bf(nb_tasks, min_delay, queue, VERBOSE);
        bf.generate();

        generated_instance_v1 end_elem = {END_ID, {}, {}};
        for (size_t i=0; i<nb_threads; i++) {
            queue->push(end_elem);
        }
    }

    static void simulator_thread(WaitingQueueV1 *queue, unsigned nb_tasks, unsigned min_delay) {
        constexpr int VERBOSE = 0;
        constexpr bool SKIP_NOT_LD_SCHEDULABLE = false;
        constexpr bool SIMU_ONLY_PERIODIC = false;
        constexpr bool SIMU_ONLY_FULL_COMPUTE = true;
        constexpr bool DEDUCE_SD = true;

        const period_vec sd(nb_tasks, min_delay);
        const period_vec rd(nb_tasks, min_delay);
        while (true) {
            auto [i, c, p] = queue->pop();

            if (i == END_ID) {
                break;
            }

            Taskset taskset(c, p, sd, rd, DEDUCE_SD);
            bool ld_schedulable = taskset.is_ld_schedulable();

            if (SKIP_NOT_LD_SCHEDULABLE && !ld_schedulable) {
                continue;
            }

            std::println("[{:8}] " KBLD "Instance:" KRST " {}", i, taskset);

            SimulationSporadic ss(taskset, SIMU_ONLY_PERIODIC, SIMU_ONLY_FULL_COMPUTE, VERBOSE);
            bool simu_worked = ss.run();

            if (ld_schedulable) {
                if (simu_worked == false) {
                    std::println(std::cerr, "[{:8}] Simulation failed with an ld-schedulable taskset", i);
                    std::abort();
                }
            } else if (simu_worked) {
                std::println("[{:8}] Simulation unexpectedly worked ({} states)", i, ss.nb_simu_states());
            }

            if (simu_worked) {
                const auto seen_rt = ss.get_worst_response_times();
                for (size_t j=0; j<taskset.size(); j++) {
                    if (seen_rt[j] < taskset.ld_response_time[j]) {
                        std::println("[{:8}] Did not see computed response time for task {}: {} < {}",
                                     i, j, seen_rt[j], taskset.ld_response_time[j]);
                        //std::abort();
                    }
                }
            }
        }
    }

    void run_simulators(WaitingQueueV1 &queue, size_t nb_threads, unsigned nb_tasks, unsigned min_delay) {
        std::vector<std::thread> s_threads;

        for (unsigned int thread_idx=0; thread_idx<nb_threads; thread_idx++) {
            s_threads.emplace_back(std::thread(simulator_thread, &queue, nb_tasks, min_delay));
        }

        std::println("Started {} simulator threads", s_threads.size());

        /* Wait for the threads to finish */
        for (auto& thread_i: s_threads) {
            thread_i.join();
        }
    }

public:
    FSimuV1(unsigned nb_tasks, unsigned min_delay, bool multithreaded) {
        size_t nb_threads = 1;
        if (multithreaded) {
            nb_threads = std::thread::hardware_concurrency();
        }

        WaitingQueueV1 queue;

        std::thread gen(generator_thread, &queue, nb_threads, nb_tasks, min_delay);

        run_simulators(queue, nb_threads, nb_tasks, min_delay);

        gen.join();
    }
};

class FSimuV2 {
    /*
        Automaton:
        Note: we only modify the lowest priority task (or add a new task)
        if period unfrozen:
            -> increment period (only case the taskset is made "easier")
            -> freeze period
        else:
            if simulation works:
                if compute unfrozen:
                    -> augment compute
                    -> freeze compute
                else:
                    if SD unfrozen:
                        -> augment SD (up to sd<c; remember we are in deduce mode) (do not do it for tau_1)
                        -> freeze SD
                    else:
                        if not (SD=0 and C=1): (othewise RD will never be run...)
                            -> augment RD (up to sd+c+rd<p) (do not do it for tau_1)
                        if utilization <1:
                            -> add new task with period unfrozen, using minimal period such that new_util<=1
                               and regular_schedulable

        WeightedWaitingQueue:
        weight: sum(cost(i) * period_i)
            cost(i) = i (or 1)

        Push to queue:
            Update 'id'
            Early not add:
                - util > 1
                - not regular_schedulable
    */

    static constexpr int VERBOSE = 0;
    static constexpr bool SKIP_NOT_LD_SCHEDULABLE = false;
    static constexpr bool SIMU_ONLY_PERIODIC = false;
    static constexpr bool SIMU_ONLY_FULL_COMPUTE = true;
    static constexpr bool DEDUCE_SD = true;

    static std::atomic<u64> next_id;

    [[gnu::pure]] inline
    static bool is_regular_schedulable(const GeneratedInstanceV2 &elem) {
        /* We can look only at the lowest-priority task. Otherwise, it would not have gone that far */
        size_t last = elem.p.size() - 1;

        period_t rn = 0;
        bool dirty = true;
        while (dirty && rn <= elem.p[last]) {
            dirty = false;
            period_t new_rn = DEDUCE_SD ? elem.c[last] : (elem.sd[last] + elem.c[last]);
            for (size_t j=0; j<last; j++) {
                period_t ceildiv = (rn + elem.p[j] - 1) / elem.p[j];
                period_t release_cost = DEDUCE_SD ? elem.c[j] : (elem.sd[j] + elem.c[j]);
                new_rn += ceildiv * release_cost;
            }
            if (rn != new_rn) {
                dirty = true;
            }
            rn = new_rn;
        }

        return rn <= elem.p[last];
    }

    inline
    static void push_instance(WeightedWaitingQueue *queue, GeneratedInstanceV2 &elem, bool is_harder) {
        auto old_id = elem.id;
        auto new_id = next_id++;

        bool utilization_less_one = true;
        bool regular_schedulable = true;
        if (is_harder) {
            auto [utilization, utilization_one] = elem.compute_utilization();
            if (utilization > utilization_one) {
                utilization_less_one = false; /* utilization > 1, skip*/
            }
            if (!is_regular_schedulable(elem)) {
                regular_schedulable = false; /* not even regular-schedulable, which is non-pessimistic*/
            }
        }

        if (VERBOSE > 0) {
            Taskset taskset(elem.c, elem.p, elem.sd, elem.rd, DEDUCE_SD);
            std::string bailout = "";
            if (!utilization_less_one) {
                bailout = "bailout: util>1";
            } else if (!regular_schedulable) {
                bailout = "bailout: not_reg_schedulable>1";
            }
            std::println("[{:8}]    PI: {:8} -> {} {}", old_id, new_id, taskset, bailout);
        }

        if ((!utilization_less_one) || (!regular_schedulable)) {
            return; /* Early bailout*/
        }

        elem.id = new_id;
        queue->push(elem);
        elem.id = old_id;
    }

    inline
    static bool do_simu(const GeneratedInstanceV2 &elem, const Taskset &taskset, bool ld_schedulable, double util_fp) {
        SimulationSporadic ss(taskset, SIMU_ONLY_PERIODIC, SIMU_ONLY_FULL_COMPUTE, VERBOSE - 1);
        bool simu_worked = ss.run();
        bool unexpected_worked = false;
        bool pessimistic = false;
        auto id = elem.id;

        if (ld_schedulable) {
            if (simu_worked == false) {
                std::println(std::cerr, "[{:8}] Simulation failed with ld-schedulable taskset: {}", id, taskset);
                std::abort();
            }
        } else if (simu_worked) {
            if (VERBOSE > 0) {
                std::println("[{:8}] Simulation unexpectedly worked ({} states)", id, ss.nb_simu_states());
            }
            unexpected_worked = true;
        }

        if (simu_worked) {
            const auto seen_rt = ss.get_worst_response_times();
            for (size_t j=0; j<taskset.size(); j++) {
                assert(seen_rt[j] <= taskset.ld_response_time[j]);
                bool seen_less = (seen_rt[j] < taskset.ld_response_time[j]);
                if (seen_less) {
                    if (VERBOSE > 0) {
                        std::println("[{:8}] Did not see computed response time for task {}: {} < {}",
                                        id, j, seen_rt[j], taskset.ld_response_time[j]);
                    }
                    pessimistic = true;
                }
            }
        }

        if (VERBOSE >= 0) {
            std::string str_seen_rt;
            const char *work_status =
                simu_worked ? (unexpected_worked ? "😱" : (pessimistic ? "🐌" : "✅")) : "❌";

            if (pessimistic) {
                if (VERBOSE >= 0) {
                    str_seen_rt = " | RT:" KMAJ "[";
                }
                const auto seen_rt = ss.get_worst_response_times();
                for (size_t j=0; j<taskset.size(); j++) {
                    bool seen_less = (seen_rt[j] < taskset.ld_response_time[j]);
                    if (VERBOSE >= 0) {
                        std::format_to(std::back_inserter(str_seen_rt), "{}{}{}{}",
                            ((j > 0) ? ", " : ""),
                            (seen_less ? KBLD : ""),
                            seen_rt[j],
                            (seen_less ? KREG : ""));
                    }
                }
                if (VERBOSE >= 0) {
                    std::format_to(std::back_inserter(str_seen_rt), "]" KRST);
                }
            }

            std::println("[{:8}] " KBLD "Instance:" KRST " util:" KYLW "{:.2f}" KRST " {} "
                        KDIM "w:{:3} ss:{:5} " KRST "| {}{}",
                         id, util_fp, work_status, elem.weight, ss.nb_simu_states(), taskset, str_seen_rt);
        }

        return simu_worked;
    }

    static void worker_thread(WeightedWaitingQueue *queue) {
        while (true) {
            auto elem = queue->pop();
            if (elem.weight == elem.END_WEIGHT) [[unlikely]] {
                /* Dark-magic which works well-enough in practice. */
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (queue->is_empty()) {
                    break;
                }
                queue->push_no_checkweight(elem);
                continue;
            }

            auto id = elem.id;
            Taskset taskset(elem.c, elem.p, elem.sd, elem.rd, DEDUCE_SD);
            bool ld_schedulable = taskset.is_ld_schedulable();

            if (SKIP_NOT_LD_SCHEDULABLE && !ld_schedulable) {
                continue;
            }

            if (VERBOSE > 0) {
                std::println("[{:8}] Will process: {}", id, taskset);
            }

            double util_fp;
            {
                auto [utilization, utilization_one] = elem.compute_utilization();
                util_fp = (double) utilization / utilization_one;
            }

            bool simu_worked = do_simu(elem, taskset, ld_schedulable, util_fp);

            if (VERBOSE > 0) {
                std::println("[{:8}]  SIMU: worked:{}", id, simu_worked);
            }

            /* Now add new elements */

            size_t last = elem.p.size() - 1;

            if (!elem.pn_frozen) {
                if (VERBOSE > 0) {
                    std::println("[{:8}]  AA: increment pn={}", id, elem.p[last]);
                }
                elem.p[last]++;
                auto old_weight = elem.weight;
                elem.add_weight_period_increment();
                push_instance(queue, elem, false);
                elem.weight = old_weight;
                elem.p[last]--;
                /* fallthrough to pn_frozen */
                elem.pn_frozen = true;
            }

            if (simu_worked) {
                if (!elem.cn_frozen) {
                    if (VERBOSE > 0) {
                        std::println("[{:8}]  AA: increment cn={}", id, elem.c[last]);
                    }
                    if (elem.c[last] < elem.p[last]) {
                        elem.c[last]++;
                        push_instance(queue, elem, true);
                        elem.c[last]--;
                    }
                    /* fallthrough to cn_frozen */
                    elem.cn_frozen = true;
                }

                if (!elem.sdn_frozen) {
                    if ((last > 0) && (elem.sd[last] + 1 < elem.c[last])) {
                        if (VERBOSE > 0) {
                            std::println("[{:8}]  AA: increment sdn={}", id, elem.sd[last]);
                        }
                        elem.sd[last]++;
                        push_instance(queue, elem, true);
                        elem.sd[last]--;
                    }
                    /* fallthrough to sdn_frozen */
                    elem.sdn_frozen = true;
                }

                if (!((elem.sd[last] == 0) && (elem.c[last] == 1))) {
                    if ((last > 0) && (elem.sd[last] + elem.c[last] + elem.rd[last] + 1 < elem.p[last])) {
                        if (VERBOSE > 0) {
                            std::println("[{:8}]  AA: increment rdn={}", id, elem.rd[last]);
                        }
                        elem.rd[last]++;
                        push_instance(queue, elem, true);
                        elem.rd[last]--;
                    }
                }

                auto [utilization, utilization_one] = elem.compute_utilization();
                if (utilization < utilization_one) {
                    /*
                     * We want to minimize 'p' such that 'utilization + 1/p <= 1'
                     * i.e. p = ceil(1/(1-utilization))
                     */
                    util_t util_left = utilization_one - utilization;
                    util_t tmp;
                    bool overflow = __builtin_uaddl_overflow(utilization_one, util_left, &tmp);
                    if (overflow) {
                        std::println(std::cerr, "Overflow in {}", __func__);
                        std::abort();
                    }
                    util_t util_ceildiv = (utilization_one + util_left - 1) / util_left;
                    if (util_ceildiv > std::numeric_limits<period_t>::max()) {
                        std::println(std::cerr, "[{:8}] Overflow in compute new task period", id);
                        std::abort();
                    }

                    period_t new_p = util_ceildiv;
                    elem.p.push_back(new_p);
                    elem.c.push_back(1);
                    elem.sd.push_back(0);
                    elem.rd.push_back(0);
                    elem.pn_frozen = false;
                    elem.cn_frozen = false;
                    elem.sdn_frozen = false;
                    last++;

                    bool good = true;
                    while (!is_regular_schedulable(elem)) {
                        if (elem.p[last] > 10'000) {
                            std::println("[{:8}] Too high period for reg_sched in new task period", id);
                            //std::abort();
                            good = false;
                            break;
                        }
                        elem.p[last]++;
                    }

                    if (good) {
                        elem.add_weight_new_task();

                        if (VERBOSE > 0) {
                            std::println("[{:8}]  AA: new_task pn:{}", id, new_p);
                        }

                        push_instance(queue, elem, true);
                    }
                    /* No need to restore elem */
                }
            }
        }
    }

public:
    FSimuV2(u64 max_weight, bool multithreaded) {
        size_t nb_threads = 1;
        if (multithreaded) {
            nb_threads = std::thread::hardware_concurrency();
        }

        WeightedWaitingQueue queue(max_weight, nb_threads);
        next_id = 0;

        /* Push initial instance */
        GeneratedInstanceV2 initial_elem = GeneratedInstanceV2(0, {1}, {1}, {0}, {0});
        push_instance(&queue, initial_elem, true);

        std::vector<std::thread> s_threads;

        std::println("Starting {} simulator threads", nb_threads);
        for (unsigned int thread_idx=0; thread_idx<nb_threads; thread_idx++) {
            s_threads.emplace_back(std::thread(worker_thread, &queue));
        }

        /* Wait for the threads to finish */
        for (auto& thread_i: s_threads) {
            thread_i.join();
        }
    }
};

class RunForGraph {
    /*
     * Steps:
     *  - for nb_tasks = 1 to ..., do NB_RUNS of
     *    - Generate a taskset (using GeneratorMulti)
     *    - Compute RiLD for all tasks
     *      - if <= Di, then all good
     *      - else, simulate the taskset in:
     *        - periodic with offset mode
     *        - periodic with a given offset tuple
     *        - periodic without offset
     *    - Print results
     *  - Build a nice graph (side python script)
     */

public:
    class Config {
    public:
        std::vector<u64> TASKS_NBS;
        std::vector<u64> UTILIZATION_BIN_MIN_CUTOFF_ARRAY;
        TG::Config taskgen_config;
    };

private:
    static constexpr u64 TARGET_ELEM_PER_UTILIZATION_BIN = 10'000;

    static constexpr int VERBOSE = 0;
    static constexpr bool SIMU_ONLY_FULL_COMPUTE = true;
    static constexpr bool CONFIG_DEDUCE_SD = false;

    const Config config;
    const bool SKIP_LD_SCHEDULABLE;
    size_t nb_tasks;
    std::atomic_flag out_f_lock;
    std::ofstream out_f;
    static constexpr size_t util_nb_bins = 101;
    std::array<std::atomic<u64>, util_nb_bins> util_bin;
    size_t util_bin_min_cutoff;
    std::atomic<unsigned> util_bin_reached;
    std::atomic<u64> gen_tries;
    std::atomic<u64> gen_kept;
    std::chrono::time_point<std::chrono::steady_clock> workers_started;

    void print_progress() {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = now - workers_started;
        std::string detail;
        for (size_t i=0; i<util_nb_bins; i++) {
            u64 progress = 10 * util_bin[i] / TARGET_ELEM_PER_UTILIZATION_BIN;
            if (i < util_bin_min_cutoff) {
                detail.append(KDIM);
            }
            if (progress < 10) {
                std::format_to(std::back_inserter(detail), "{}", progress);
            } else {
                detail.append(KGRN "*");
            }
            detail.append(KRST);
        }
        auto read_util_bin_reached = util_bin_reached.load();
        auto read_tries = gen_tries.load();
        std::println(KBLD "GeneratorProgress" KRST ": " KYLW "{:3}/{}" KRST ": {} tries:{:12}" KDIM " ({}s)" KRST,
                     read_util_bin_reached, util_nb_bins, detail, read_tries, elapsed.count());
    }

    bool generate(TG &tg, TG::GeneratedInstance &elem) {
        bool ok = false;

        constexpr std::array<u64, 5> progress_itv_array = {0x4000, 0x4000, 0x4000, 0x4000, 0x1000};
        u64 progress_itv = Utils::get_val_or_last(progress_itv_array, nb_tasks);
        while ((!ok) && (util_bin_reached < util_nb_bins)) {
            gen_tries++;
            tg.generate(elem);

            size_t util_idx = (size_t)((util_nb_bins - 1) * elem.utilization);
            assert (util_idx < util_nb_bins);
            if (util_bin[util_idx] < TARGET_ELEM_PER_UTILIZATION_BIN) {
                ok = true;
                auto kept = gen_kept++;

                auto new_util_bin_count = ++util_bin[util_idx];
                if (new_util_bin_count == TARGET_ELEM_PER_UTILIZATION_BIN) {
                    if (util_idx >= util_bin_min_cutoff) {
                        util_bin_reached++;
                    }
                }

                if ((kept % progress_itv) == 0) {
                    print_progress();
                }
            }
        }

        return ok;
    }

    void log_instance(const TG::GeneratedInstance &elem, const Taskset &taskset, const SimulationSporadic *ss,
                      bool ld_schedulable, bool simu_worked) {
        std::string log_line;
        std::ostringstream all_ri;
        std::ostringstream all_rild;
        for (size_t j=0; j<taskset.size(); j++) {
            if (j > 0) {
                all_ri << ",";
                all_rild << ",";
            }
            auto rild = taskset.compute_rild(j, true, true);
            auto ri = taskset.compute_rild(j, false, true);
            all_ri << ri;
            all_rild << rild;
        }

        std::ostringstream all_rti;
        if (ss != nullptr) {
            const auto seen_rt = ss->get_worst_response_times();
            for (size_t j=0; j<taskset.size(); j++) {
                if (j > 0) {
                    all_rti << ",";
                }
                all_rti << seen_rt[j];
            }
        }

        char print_simutype;
        switch (config.taskgen_config.simu_type) {
            case TG::SimuTypePeriodicGivenOffset:
                print_simutype = 'P';
                break;
            case TG::SimuTypePeriodicAnyOffset:
                print_simutype = 'O';
                break;
            case TG::SimuTypeSporadic:
                print_simutype = 'S';
                break;
            default:
                std::println(std::cerr, "Bad simutype");
                std::abort();
        }

        std::format_to(std::back_inserter(log_line), "{};{};{};{};{};{};{};{};{};{};{};{}\n",
            nb_tasks, print_simutype,
            config.taskgen_config.max_period, Utils::print_bool(config.taskgen_config.sd_geq_rd, false),
            elem.utilization, taskset.compute_badldutil(), taskset.compute_ldutil(),
            Utils::print_bool(ld_schedulable, false), Utils::print_bool(simu_worked, false),
            all_ri.str(), all_rild.str(), all_rti.str()
        );

        while (out_f_lock.test_and_set(std::memory_order_acquire)) {}
        out_f << log_line;
        out_f_lock.clear();
    }

    void worker_thread(unsigned thread_idx) {
        u64 seed = thread_idx;
        TaskGenerator tg = TaskGenerator(nb_tasks, seed, config.taskgen_config, VERBOSE);

        bool simu_only_periodic = false;
        bool set_offsets = false;
        switch (config.taskgen_config.simu_type) {
            case TG::SimuTypePeriodicGivenOffset:
                simu_only_periodic = true;
                set_offsets = true;
                break;
            case TG::SimuTypePeriodicAnyOffset:
                simu_only_periodic = true;
                break;
            case TG::SimuTypeSporadic:
                break; /* Nothing needed */
            default:
                std::println(std::cerr, "Bad simutype");
                std::abort();
        }

        for (u64 i=0; /* none */; i++) {
            TG::GeneratedInstance elem(i, nb_tasks);
            bool ok = generate(tg, elem);
            if (!ok) {
                break;
            }

            Taskset taskset(elem.c, elem.p, elem.sd, elem.rd, CONFIG_DEDUCE_SD);
            bool ld_schedulable = taskset.is_ld_schedulable();
            bool simu_worked;
            size_t simu_states = 0;

            if (SKIP_LD_SCHEDULABLE && ld_schedulable) {
                simu_worked = true; /* By definition */
                log_instance(elem, taskset, nullptr, ld_schedulable, simu_worked);
            } else {
                SimulationSporadic ss(taskset, simu_only_periodic, SIMU_ONLY_FULL_COMPUTE, VERBOSE);
                if (set_offsets) {
                    ss.set_offsets(elem.o);
                }
                simu_worked = ss.run();
                simu_states = ss.nb_simu_states();
#if 0 //TODO
                double ldratio = 0.;
                if (simu_worked) {
                    auto t1 = std::chrono::steady_clock::now();
                    ldratio = ss.compute_ldratio();
                    auto t2 = std::chrono::steady_clock::now();
                    std::chrono::duration<float> e21 = t2 - t1;
                    static double timeacc = 0.;
                    timeacc += e21.count();
                    static std::atomic<int> todo = 0;
                    todo += 1;
                    std::println("[{: 6}] Ts:{} simu_states:{: 6} ldratio:{:.2f} " KDIM "({:.3f}s | {:.3f}s)" KRST,
                                 todo.load(), taskset, simu_states, ldratio, e21.count(), timeacc);
                }
#endif

                log_instance(elem, taskset, &ss, ld_schedulable, simu_worked);
            }

            if (VERBOSE > 0) {
                std::println("[{:8}] " KBLD "R4G-Instance:" KRST " LD_SCHED:{} SIMU_WORKED:{} "
                             "util:" KYLW "{:.2f}" KRST " {} ({} states)",
                             i, Utils::print_bool(ld_schedulable), Utils::print_bool(simu_worked),
                             elem.utilization, taskset, simu_states);
            }

        }
    }

    void run_workers(size_t nb_threads) {
        /* Reset bins */
        util_bin_min_cutoff = config.UTILIZATION_BIN_MIN_CUTOFF_ARRAY.back();
        if (nb_tasks < config.UTILIZATION_BIN_MIN_CUTOFF_ARRAY.size()) {
            util_bin_min_cutoff = config.UTILIZATION_BIN_MIN_CUTOFF_ARRAY[nb_tasks];
        }
        util_bin_reached = util_bin_min_cutoff;
        std::fill(util_bin.begin(), util_bin.end(), 0);
        gen_tries = 0;
        gen_kept = 0;


        std::println("Starting {} simulator threads for {} tasks", nb_threads, nb_tasks);
        std::vector<std::thread> s_threads;
        workers_started = std::chrono::steady_clock::now();

        for (unsigned int thread_idx=0; thread_idx<nb_threads; thread_idx++) {
            s_threads.emplace_back(std::thread([&] (unsigned idx) { this->worker_thread(idx); }, thread_idx));
        }

        /* Wait for the threads to finish */
        for (auto& thread_i: s_threads) {
            thread_i.join();
        }

        print_progress(); /* Print the nice "completed" progress status */
    }

    public:
    RunForGraph(const Config &_config, const std::string &out_file,
                bool _skip_ld_schedulable, bool multithreaded)
        : config(_config), SKIP_LD_SCHEDULABLE(_skip_ld_schedulable) {
        size_t nb_threads_system = 1;
        unsigned long totalram_mib = 0;
        if (multithreaded) {
            nb_threads_system = std::thread::hardware_concurrency();

            struct sysinfo info;
            int ret = sysinfo(&info);
            if (ret != 0) {
                std::println(std::cerr, "Sysinfo failed");
                std::abort();
            }
            totalram_mib = info.totalram / (1024*1024);
        }

        constexpr std::string_view out_dir = "out";
        std::filesystem::create_directory(out_dir);
        out_f.open(std::string(out_dir) + "/" + out_file);
        out_f_lock.clear();

        constexpr std::array<u64, 6> ram_by_nb_tasks_spo = {1, 1, 1, 800, 4'000, 200'000}; /* In MiB */
        constexpr std::array<u64, 6> ram_by_nb_tasks_per = {1, 1, 1,  10,   100,     100}; /* In MiB */
        static_assert(TG::SimuTypeSporadic == 0);
        static_assert(TG::SimuTypePeriodicAnyOffset == 1);
        static_assert(TG::SimuTypePeriodicGivenOffset == 2);
        const std::array ram_by_nb_tasks = {ram_by_nb_tasks_spo, ram_by_nb_tasks_per, ram_by_nb_tasks_per};

        for (auto _nb_tasks : config.TASKS_NBS) {
            nb_tasks = _nb_tasks;
            u64 ram_usage = Utils::get_val_or_last(ram_by_nb_tasks[config.taskgen_config.simu_type], nb_tasks);
            size_t nb_threads_by_ram = totalram_mib / ram_usage;
            size_t nb_threads = std::max(1zu, std::min(nb_threads_system, nb_threads_by_ram));
            run_workers(nb_threads);
        }
    }
};

std::atomic<u64> FSimuV2::next_id{0}; /* Cannot be declared inside the class */

static void test_single() {
    /* c, p, sd, rd */
    Taskset taskset({1,1}, {10,8}, {0,0}, {0,0}, false);
    std::println("[single] " KBLD "Instance:" KRST " {}", taskset);

    bool only_periodic = false;
    bool only_full_compute = false;
    bool verbose = 1;

    SimulationSporadic ss(taskset, only_periodic, only_full_compute, verbose);
    bool simu_worked = ss.run();
    std::println("[single] simu_worked:{}", simu_worked);
    if (simu_worked) {
        double lambda = ss.compute_ldratio();
        std::println("[single] lambda:{}", lambda);
    }
}

int main(int argc, char *argv[]) {
    constexpr unsigned DEFAULT_VERSION = 4;
    constexpr bool MULTITHREADED = true;
    /* For V1 */
    constexpr unsigned NB_TASKS = 3;
    constexpr unsigned MIN_DELAY = 0;
    /* For V2 */
    constexpr u64      MAX_WEIGHT = 1'000;
    /* For V3-5 */
    constexpr u64 MAX_PERIOD = 100;
    [[gnu::unused]] constexpr bool SKIP_LD_SCHEDULABLE = true;
    constexpr bool RUN_LD_SCHEDULABLE = false;
    RunForGraph::Config config_paper_vs_periodic(
        {2, 3, 4, 6, 8, 16}, /* Nb tasks */
        {0, 0, 2, 3, 4, 5, 6, 10, /*8t*/ 10, 25, 25, 25, 25, 25, 25, 25, /*16*/25 }, /* Util cutoff */
        TG::Config(MAX_PERIOD, TG::SimuTypePeriodicGivenOffset, TG::CONFIG_SD_GEQ_RD, TG::PeriodGeneratorSmallHP));
    RunForGraph::Config config_paper_spo_pess(
        {2, 3, 4}, /* Nb tasks */
        {0, 0, 2, 3, 4}, /* Util cutoff */
        TG::Config(MAX_PERIOD, TG::SimuTypeSporadic           , TG::CONFIG_ANY_SD,    TG::PeriodGeneratorSmallHP));

    unsigned version = DEFAULT_VERSION;
    if (argc >= 2) {
        version = std::stoi(argv[1]);
    }

    switch (version) {
        case 0:
            test_single();
            break;
        case 1:
            FSimuV1(NB_TASKS, MIN_DELAY, MULTITHREADED);
            break;
        case 2:
            FSimuV2(MAX_WEIGHT, MULTITHREADED);
            break;
        case 3:
            RunForGraph(config_paper_vs_periodic, "out_pergiven_shp.txt", RUN_LD_SCHEDULABLE, MULTITHREADED);
            break;
        case 4:
            RunForGraph(config_paper_spo_pess,    "out_spo_shp.txt",      RUN_LD_SCHEDULABLE, MULTITHREADED);
            break;
        default:
            std::println(std::cerr, "Unknown version {}", version);
            std::abort();
    }
}
