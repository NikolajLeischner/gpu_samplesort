#pragma once

struct KeyType {
	enum class Value { uint16_t, uint32_t, uint64_t };

	static Value parse(std::string value);
};

