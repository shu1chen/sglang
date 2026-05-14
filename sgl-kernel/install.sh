export VERBOSE=1
export http_proxy=http://proxy-dmz.intel.com:911
export https_proxy=http://proxy-dmz.intel.com:912

cp pyproject_cpu.toml pyproject.toml
uv pip install -v . --no-build-isolation
git checkout pyproject.toml

