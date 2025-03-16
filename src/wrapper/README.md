# [`src/wrapper`](./): directory for the customized Environment Wrapper used in this project

This directory contains the Wrappers implementation of the project. Below is a detailed overview of each module and its purpose.

## Directory Structure

- [`calf_wrapper.py`](./calf_wrapper.py) - CALF related components
  - CALF Environment Wrapper and its variants
  - Modules used in the experiemnt [`ppo_pendulum_calf_wrapper_eval`](../../run/ppo_pendulum_calf_wrapper_eval):
    - **CALF_Wrapper**: This CALF wrapper filter use exponential Relax probability decay
    - **CALFEnergyPendulumWrapper**: A firm layer for CALF fallback to get action from EnergyBasedController
  - Modules used in the experiemnt [`ppo_vispendulum_self_boost`](../../run/ppo_vispendulum_self_boost):
    - **CALFWrapperCustomizedRelaxProb**: This CALF wrapper filter use `RelaxProb` decay
    - **CALFPPOPendulumWrapper**(CALFNominalWrapper): A firm layer for CALF fallback to get action from a checkpoint of PPO (defined in [`Python script`](../../run/ppo_vispendulum_self_boost/ppo_vispendulum_eval_calf_wrapper.py))
    - **RelaxProb**: Support linear decay of Relax Probability
- [`common_wrapper.py`](./common_wrapper.py) - A customized Environment Wrappers
  - Standard Pendulum Environment Wrappers
  - Pendulum Environment Wrappers with visual observation

## Contributing Guidelines

### Code Modifications
- Add new core functionality to the [`src/wrapper/`](./) directory
- When modifying existing code, ensure backwards compatibility
- Verify that existing experiments and results remain valid
- Document all significant changes
