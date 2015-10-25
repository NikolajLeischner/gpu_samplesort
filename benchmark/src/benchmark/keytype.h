#pragma once

struct KeyType {
	enum class Value { uint8_t, uint16_t, uint32_t, uint64_t, uint128_t, float_t, double_t };

	static Value parse(std::string value);
};

