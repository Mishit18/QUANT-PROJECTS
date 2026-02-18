#include "memory_pool.h"
#include "order.h"

namespace lob {

// Explicit template instantiation
template class MemoryPool<Order, MAX_ORDERS>;

} // namespace lob
