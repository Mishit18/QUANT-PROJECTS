#include "order_book.h"
#include <algorithm>

namespace lob {

OrderBook::OrderBook() noexcept
    : num_active_bids_(0), num_active_asks_(0),
      best_bid_price_(INVALID_PRICE), best_ask_price_(INVALID_PRICE),
      best_bid_level_(nullptr), best_ask_level_(nullptr) {
    
    for (size_t i = 0; i < MAX_PRICE_LEVELS; ++i) {
        active_bids_[i] = nullptr;
        active_asks_[i] = nullptr;
    }
}

void OrderBook::add_order(Order* order) noexcept {
    PriceLevel* level = get_or_create_level(order->price, order->side);
    level->add_order(order);
    
    if (order->is_buy()) {
        if (best_bid_price_ == INVALID_PRICE || order->price > best_bid_price_) {
            best_bid_price_ = order->price;
            best_bid_level_ = level;
        }
    } else {
        if (best_ask_price_ == INVALID_PRICE || order->price < best_ask_price_) {
            best_ask_price_ = order->price;
            best_ask_level_ = level;
        }
    }
}

void OrderBook::remove_order(Order* order) noexcept {
    PriceLevel* level = get_or_create_level(order->price, order->side);
    level->remove_order(order);
    
    if (level->empty()) {
        if (order->is_buy() && level == best_bid_level_) {
            update_best_bid();
        } else if (order->is_sell() && level == best_ask_level_) {
            update_best_ask();
        }
    }
}

void OrderBook::update_order(Order* order, Quantity old_remaining) noexcept {
    PriceLevel* level = get_or_create_level(order->price, order->side);
    level->update_quantity(order, old_remaining);
    
    if (order->is_filled()) {
        level->remove_order(order);
        if (level->empty()) {
            if (order->is_buy() && level == best_bid_level_) {
                update_best_bid();
            } else if (order->is_sell() && level == best_ask_level_) {
                update_best_ask();
            }
        }
    }
}

PriceLevel* OrderBook::best_bid() noexcept {
    return best_bid_level_;
}

PriceLevel* OrderBook::best_ask() noexcept {
    return best_ask_level_;
}

bool OrderBook::can_match_buy(Price price) const noexcept {
    return best_ask_price_ != INVALID_PRICE && price >= best_ask_price_;
}

bool OrderBook::can_match_sell(Price price) const noexcept {
    return best_bid_price_ != INVALID_PRICE && price <= best_bid_price_;
}

PriceLevel* OrderBook::get_or_create_level(Price price, Side side) noexcept {
    if (side == Side::BUY) {
        for (size_t i = 0; i < num_active_bids_; ++i) {
            if (active_bids_[i]->price() == price) {
                return active_bids_[i];
            }
        }
        size_t idx = num_active_bids_++;
        active_bids_[idx] = &bid_levels_[idx];
        active_bids_[idx]->initialize(price);
        return active_bids_[idx];
    } else {
        for (size_t i = 0; i < num_active_asks_; ++i) {
            if (active_asks_[i]->price() == price) {
                return active_asks_[i];
            }
        }
        size_t idx = num_active_asks_++;
        active_asks_[idx] = &ask_levels_[idx];
        active_asks_[idx]->initialize(price);
        return active_asks_[idx];
    }
}

void OrderBook::update_best_bid() noexcept {
    best_bid_price_ = INVALID_PRICE;
    best_bid_level_ = nullptr;
    
    for (size_t i = 0; i < num_active_bids_; ++i) {
        if (!active_bids_[i]->empty()) {
            Price p = active_bids_[i]->price();
            if (best_bid_price_ == INVALID_PRICE || p > best_bid_price_) {
                best_bid_price_ = p;
                best_bid_level_ = active_bids_[i];
            }
        }
    }
}

void OrderBook::update_best_ask() noexcept {
    best_ask_price_ = INVALID_PRICE;
    best_ask_level_ = nullptr;
    
    for (size_t i = 0; i < num_active_asks_; ++i) {
        if (!active_asks_[i]->empty()) {
            Price p = active_asks_[i]->price();
            if (best_ask_price_ == INVALID_PRICE || p < best_ask_price_) {
                best_ask_price_ = p;
                best_ask_level_ = active_asks_[i];
            }
        }
    }
}

} // namespace lob
