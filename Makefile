test:
	cd tests && python -m pytest . -s --disable-warnings

lint:
	python -m pylint cipherface/ --fail-under=10