# Example usage and experiment runner
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from environment import PrisonersDilemmaEnvironment

class ExperimentRunner:
    """Helper class to run and analyze experiments"""
    
    def __init__(self, llm_pipeline):
        self.llm_pipeline = llm_pipeline
    
    def run_experiment(self, population_size: int = 30, num_generations: int = 5,
                      mutation_prob: float = 0.05, noise_prob: float = 0.05) -> Dict:
        """Run a complete experiment"""
        
        # Create environment
        env = PrisonersDilemmaEnvironment(
            llm_pipeline=self.llm_pipeline,
            population_size=population_size,
            mutation_prob=mutation_prob,
            noise_prob=noise_prob
        )
        
        # Initialize population
        env.initialize_population()
        
        # Run evolution
        results = env.run_evolution(num_generations, verbose=True)
        
        # Analyze results
        word_analysis = env.analyze_population_words()
        
        # Visualize
        env.visualize_evolution()
        
        # Print summary
        final_stats = results['final_stats']
        print(f"\n=== Experiment Summary ===")
        print(f"Final cooperation rate: {final_stats['avg_cooperation']:.3f} ± {final_stats['std_cooperation']:.3f}")
        print(f"Final fitness: {final_stats['avg_fitness']:.3f} ± {final_stats['std_fitness']:.3f}")
        
        print("\nSample evolved personality genes:")
        for i, agent in enumerate(results['final_population'][:5]):
            print(f"{i+1}. {agent.personality_gene}")
        
        print(f"\nTop words by cooperation rate:")
        sorted_words = sorted(word_analysis.items(), 
                            key=lambda x: x[1]['avg_cooperation'], reverse=True)
        for word, stats in sorted_words[:10]:
            print(f"{word}: coop={stats['avg_cooperation']:.3f}, freq={stats['frequency']}")
        
        return {
            'environment': env,
            'evolution_results': results,
            'word_analysis': word_analysis
        }
        

model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
print("--- LLM Pipeline created. Initializing Experiment Runner. ---")
runner = ExperimentRunner(pipe)

results = runner.run_experiment(
    population_size=30,      # N in paper
    num_generations=5,    # G in paper
    mutation_prob=0.05,      # pm in paper
    noise_prob=0.05          # pn in paper
)

print("\n--- Experiment Finished ---")