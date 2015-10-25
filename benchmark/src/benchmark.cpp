#include <iostream>
#include <distributions.h>
#include <settings.h>
#include <tclap/CmdLine.h>

int main(int argc, char *argv[]) 
{  
	try {

		auto settings = Benchmark::Settings::parse_from_cmd(argc, argv);

		return 0;
	}
	catch (TCLAP::ArgException &e) {
		std::cerr << "error: " << e.error() << " for argument " << e.argId() << std::endl;
		return 1;
	}
}

