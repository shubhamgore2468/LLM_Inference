// cpp/src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "block.h"
#include "scheduler.h"

namespace py = pybind11;
using namespace kvsched;

// Zero-copy view over the vector's memory. The `base` keepalive ties the
// array's lifetime to the owning BatchPlan — without it Python can hold a
// dangling pointer after the plan is destroyed. ASan catches this; a release
// build silently returns garbage.
template <typename T>
static py::array_t<T> view(const std::vector<T>& v, py::object base) {
    return py::array_t<T>({static_cast<py::ssize_t>(v.size())},
                          {sizeof(T)}, v.data(), base);
}

PYBIND11_MODULE(_kvsched, m) {
    py::class_<BlockManager>(m, "BlockManager")
        .def(py::init<int32_t, int32_t>(), py::arg("n_blocks"), py::arg("block_size"))
        .def("num_free", &BlockManager::num_free)
        .def("num_total", &BlockManager::num_total)
        .def("utilization", &BlockManager::utilization);

    py::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<>())
        .def_readwrite("block_size", &SchedulerConfig::block_size)
        .def_readwrite("max_batch_seqs", &SchedulerConfig::max_batch_seqs)
        .def_readwrite("max_batch_tokens", &SchedulerConfig::max_batch_tokens)
        .def_readwrite("chunk_size", &SchedulerConfig::chunk_size)
        .def_readwrite("max_kv_utilization", &SchedulerConfig::max_kv_utilization);

    py::class_<Stats>(m, "Stats")
        .def_readonly("waiting", &Stats::waiting)
        .def_readonly("running", &Stats::running)
        .def_readonly("in_flight_tokens", &Stats::in_flight_tokens)
        .def_readonly("total_finished", &Stats::total_finished)
        .def_readonly("total_preempted", &Stats::total_preempted)
        .def_readonly("prefix_hit_blocks", &Stats::prefix_hit_blocks)
        .def_readonly("prefill_blocks", &Stats::prefill_blocks)
        .def_readonly("kv_util", &Stats::kv_util);

    py::class_<BatchPlan>(m, "BatchPlan")
        .def_readonly("is_prefill", &BatchPlan::is_prefill)
        .def_readonly("max_blocks", &BatchPlan::max_blocks)
        .def_readonly("max_seqlen_q", &BatchPlan::max_seqlen_q)
        .def_readonly("max_seqlen_k", &BatchPlan::max_seqlen_k)
        .def("n_seqs", &BatchPlan::n_seqs)
        .def("n_tokens", &BatchPlan::n_tokens)
        .def_property_readonly("seq_ids", [](py::object self) {
            return view(self.cast<const BatchPlan&>().seq_ids, self); })
        .def_property_readonly("token_ids", [](py::object self) {
            return view(self.cast<const BatchPlan&>().token_ids, self); })
        .def_property_readonly("positions", [](py::object self) {
            return view(self.cast<const BatchPlan&>().positions, self); })
        .def_property_readonly("slot_mapping", [](py::object self) {
            return view(self.cast<const BatchPlan&>().slot_mapping, self); })
        .def_property_readonly("context_lens", [](py::object self) {
            return view(self.cast<const BatchPlan&>().context_lens, self); })
        .def_property_readonly("block_tables", [](py::object self) {
            return view(self.cast<const BatchPlan&>().block_tables, self); })
        // cu_seqlens stay as Python lists: they are short, and attention.py
        // indexes them element-wise in a loop.
        .def_readonly("cu_seqlens_q", &BatchPlan::cu_seqlens_q)
        .def_readonly("cu_seqlens_k", &BatchPlan::cu_seqlens_k);

    py::class_<Scheduler>(m, "Scheduler")
        .def(py::init<BlockManager&, SchedulerConfig>(),
             py::keep_alive<1, 2>())   // scheduler must not outlive the manager
        .def("add", &Scheduler::add,
             py::arg("tokens"), py::arg("max_tokens"), py::arg("eos"),
             py::arg("ignore_eos") = false, py::arg("arrival_ts") = 0.0)
        .def("step", &Scheduler::step)
        .def("commit", &Scheduler::commit)
        .def("has_work", &Scheduler::has_work)
        .def("stats", &Scheduler::stats)
        .def("take_finished", &Scheduler::take_finished)
        .def("tokens_of", &Scheduler::tokens_of);
}
