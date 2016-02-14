#include <algorithm>
#include <map>
#include <iostream>
#include <keytype.h>

KeyType::Value KeyType::parse(std::string value) {

	std::transform(value.begin(), value.end(), value.begin(), tolower);

	std::map<std::string, KeyType::Value> types{
		{ "uint16", KeyType::Value::uint16_t },
		{ "uint32", KeyType::Value::uint32_t },
		{ "uint64", KeyType::Value::uint64_t }};

	auto result = types.find(value);
	if (result == types.end())
		throw new std::exception();
	else
		return result->second;
}