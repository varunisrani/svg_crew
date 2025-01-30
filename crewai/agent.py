class Agent:
    def __init__(self, role, goal, backstory, llm, verbose=False, allow_delegation=True):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.verbose = verbose
        self.allow_delegation = allow_delegation

    def execute(self, prompt):
        """
        Executes a prompt using the configured LLM.
        """
        try:
            response = self.llm.generate(prompt)
            if self.verbose:
                logging.info(f"Agent Response: {response}")
            return response
        except Exception as e:
            logging.error(f"Error executing prompt: {e}")
            raise 