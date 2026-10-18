from setuptools import find_packages, setup
from glob import glob

setup(name='rag_rrt_real_validation', version='1.0.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages', ['resource/rag_rrt_real_validation']),
                  ('share/rag_rrt_real_validation', ['package.xml']),
                  ('share/rag_rrt_real_validation/launch', glob('launch/*.launch.py')),
                  ('share/rag_rrt_real_validation/config', glob('config/*.yaml') + glob('config/*.rviz'))],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='RAG-RRT validation', maintainer_email='maintainer@example.com',
      description='Single task navigation validation with bundled RAG planners',
      license='Apache-2.0',
      entry_points={'console_scripts': ['trial = rag_rrt_real_validation.runner:main']})
