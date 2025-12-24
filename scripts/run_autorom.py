import sys
print('sys.executable:', sys.executable)
print('sys.path:', sys.path)
try:
	try:
		from AutoROM import AutoROM
	except Exception:
		from autorom import AutoROM
except Exception as e:
	print('import autorom failed:', repr(e))
	raise

print('running autorom...')
print('AutoROM module contents:', dir(AutoROM))
try:
	# Try common entrypoints
	if hasattr(AutoROM, 'AutoROM'):
		AutoROM.AutoROM().install_roms()
	elif hasattr(AutoROM, 'install_roms'):
		AutoROM.install_roms()
	elif hasattr(AutoROM, 'main'):
		# call main() which acts as CLI entrypoint: pass accept_license=True
		AutoROM.main(True, None, False)
	elif hasattr(AutoROM, 'cli'):
		AutoROM.cli()
	else:
		# fallback: attempt to call as function (some versions)
		try:
			AutoROM().install_roms()
		except Exception as e:
			print('Could not invoke AutoROM:', e)
			raise
except Exception as e:
    print('autorom invocation failed:', repr(e))
    raise
print('done')
