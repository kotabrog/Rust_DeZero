run:
	cargo run

test:
	cargo test

test_kdezero:
	cargo test -p kdezero

test_ktensor:
	cargo test -p ktensor

test_integrate_step:
	cargo test step -- --nocapture --test-threads=1

doc:
	cargo doc --open

dot:
	dot kdezero/output/sample.dot -T png -o kdezero/output/sample.png
