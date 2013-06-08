# Call it with a list of files to categorize in groups as arguments.
function(create_source_groups FILES)
	# Setup base rules or exit, if auto grouping is disabled.
	if (_AutoGroup)
		source_group("Auto" REGULAR_EXPRESSION "[/\\]moc_[^/\\]+$")
		source_group("CMake" REGULAR_EXPRESSION "(CMakeLists.txt)|([/\\]moc_[^/\\]+\\.rule$)")
	else (_AutoGroup)
		return()
	endif (_AutoGroup)

	# Process each given file.
	foreach(file ${ARGV})
		set(group "")
		set(file_copy ${file})
		
		if (_AutoGroup_Split)
			# Put source files in group "source".
			string(REGEX MATCH "(^src[/\\])|(.cpp$)|(.cxx$)|(.c$)" chunk ${file})
			if (chunk)
				set(group "Source")
			endif (chunk)
			
			# Put header files in group "include".
			string(REGEX MATCH "(^include[/\\])|(.hpp$)|(.hxx$)|(.h$)" chunk ${file})
			if (chunk)
				set(group "Include")
			endif (chunk)
		endif (_AutoGroup_Split)
		
		# Strip of src or include at the beginning of the file name.
		string(REGEX MATCH "^(src|include)[/\\]" chunk ${file})
		if (chunk)
			string(REGEX REPLACE "^(src|include)[/\\](.*)$" "\\2" file ${file})
		endif (chunk)
		
		# If there is any sub-path remaining.
		string(REGEX MATCH "^(.*)[/\\][^/\\]+$" chunk ${file})
		if (chunk)
			# Strip it off and append it to the group name.
			string(REGEX REPLACE "^(.*)[/\\][^/\\]+$" "\\1" chunk ${file})
			if (group)
				set(group "${group}\\")
			endif (group)
			set(group "${group}${chunk}")
		endif (chunk)
		
		# Convert path separators to nested group separators.
		if (group)
			string(REPLACE "/" "\\" group ${group})
		endif (group)
		
		# Register the file with the group.
		source_group("${group}" FILES ${file_copy})

	endforeach(file)
endfunction(create_source_groups)
