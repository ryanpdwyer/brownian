.PHONY:
	clean

clean:
	rm -rf *.pyc
	rm -rf *.pyd
	rm -rf *.pyo
	rm MANIFEST
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

