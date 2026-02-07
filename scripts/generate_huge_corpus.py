import csv
import random

# A dictionary of words with strictly conflicting meanings in different domains.
# Structure: {Word: {Domain: [Facts...]}}
AMBIGUOUS_DATA = {
    "Amazon": {
        "Tech": [
            "Amazon Web Services provides cloud computing APIs.",
            "Amazon Prime offers expedited shipping on retail goods.",
            "The Amazon Echo is a smart speaker powered by Alexa.",
            "Jeff Bezos founded Amazon in a garage in Bellevue."
        ],
        "Geography": [
            "The Amazon River discharges more water than any other river.",
            "The Amazon rainforest produces 20% of the world's oxygen.",
            "Deforestation in the Amazon threatens biodiversity.",
            "The Amazon basin covers 40% of South America."
        ]
    },
    "Python": {
        "CS": [
            "Python uses significant whitespace for indentation.",
            "Pandas and NumPy are popular Python libraries.",
            "Python is an interpreted, high-level programming language.",
            "The Global Interpreter Lock (GIL) limits Python threads."
        ],
        "Biology": [
            "Pythons are non-venomous constricting snakes.",
            "The reticulated python is the world's longest snake.",
            "Pythons swallow their prey whole after suffocation.",
            "Ball pythons are popular pets due to their docility."
        ]
    },
    "Apple": {
        "Tech": [
            "Apple Inc. designs the iPhone and MacBook hardware.",
            "The M1 chip silicon transformed Apple's computer line.",
            "iOS is the operating system for Apple mobile devices.",
            "Steve Jobs unveiled the first Apple Macintosh in 1984."
        ],
        "Fruit": [
            "The apple is a pome fruit from the Malus domestica tree.",
            "Granny Smith apples are known for their tart flavor.",
            "Apples are rich in fiber and Vitamin C.",
            "One bad apple can spoil the whole barrel."
        ]
    },
    "Jaguar": {
        "Auto": [
            "The Jaguar E-Type is a classic British sports car.",
            "Jaguar Land Rover creates luxury SUVs and sedans.",
            "The Jaguar F-Type features a supercharged V8 engine.",
            "Vintage Jaguars are highly prized by collectors."
        ],
        "Biology": [
            "The jaguar is the largest cat species in the Americas.",
            "Jaguars have a powerful bite that can pierce turtle shells.",
            "Unlike most cats, jaguars enjoy swimming in water.",
            "The jaguar's coat features rosettes with spots inside."
        ]
    },
    "Mouse": {
        "Tech": [
            "A computer mouse controls the cursor on a GUI.",
            "Gaming mice have high DPI sensors for precision.",
            "Douglas Engelbart invented the computer mouse in 1964.",
            "Wireless mice use Bluetooth or RF receivers."
        ],
        "Biology": [
            "The house mouse is a small mammal of the order Rodentia.",
            "Mice have a gestation period of only 20 days.",
            "Field mice are prey for owls, hawks, and snakes.",
            "Mice are often used in genetic research laboratories."
        ]
    },
    "Virus": {
        "Tech": [
            "A computer virus replicates by modifying other programs.",
            "Ransomware is a virus that encrypts user data for payment.",
            "Antivirus software scans files for malicious signatures.",
            "The ILOVEYOU virus infected millions of PCs in 2000."
        ],
        "Biology": [
            "A virus consists of genetic material inside a protein coat.",
            "Viruses require a living host cell to replicate.",
            "Antibiotics are ineffective against viral infections.",
            "Vaccines train the immune system to recognize viruses."
        ]
    },
    "Shell": {
        "Tech": [
            "The shell is a command-line interface for the OS.",
            "Bash and Zsh are popular Unix shells.",
            "PowerShell provides task automation on Windows.",
            "Shell scripting automates repetitive terminal commands."
        ],
        "Biology": [
            "Turtles have a protective shell fused to their spine.",
            "Seashells are the exoskeletons of marine mollusks.",
            "Hermit crabs scavenge empty shells for protection.",
            "The shell is composed primarily of calcium carbonate."
        ],
        "Energy": [
            "Shell is a global group of energy and petrochemical companies.",
            "Shell operates offshore drilling rigs in the North Sea.",
            "The Shell logo is known as the Pecten.",
            "Shell is investing in hydrogen and renewable energy."
        ]
    },
    "Chip": {
        "Tech": [
            "A microchip contains billions of transistors.",
            "Silicon chips are manufactured in fabrication plants (fabs).",
            "The shortage of chips impacted global car production.",
            "System-on-a-Chip (SoC) integrates CPU and GPU."
        ],
        "Food": [
            "Potato chips are thin slices of deep-fried potato.",
            "Chocolate chips are used in baking cookies.",
            "Fish and chips is a traditional British dish.",
            "Tortilla chips are made from corn and served with salsa."
        ]
    },
    "Stream": {
        "Tech": [
            "Video streaming buffers data over the internet.",
            "Twitch is a popular platform for live game streaming.",
            "Data streams process real-time analytics events.",
            "Streaming services have replaced physical media rental."
        ],
        "Nature": [
            "A stream is a body of water with a current.",
            "Trout often swim upstream to spawn.",
            "Streams merge to form larger rivers.",
            "The stream flow rate increases after heavy rainfall."
        ]
    },
    "Table": {
        "Database": [
            "A database table consists of rows and columns.",
            "SQL JOINs combine data from multiple tables.",
            "Primary keys ensure uniqueness in a table row.",
            "Pivot tables summarize large datasets in Excel."
        ],
        "Furniture": [
            "A dining table is where meals are served.",
            "Coffee tables are low tables placed in living rooms.",
            "Table legs provide stability and support.",
            "Set the table with forks on the left."
        ]
    }
}

NOISE_SENTENCES = [
    "The inherent variability of the system requires analysis.",
    "Data throughput is a critical metric for performance.",
    "The rapid expansion of the sector caused market volatility.",
    "Strategic alignment is necessary for long-term growth.",
    "The quantitative results were inconclusive at best.",
    "Theoretical frameworks provide a basis for understanding.",
    "Operational efficiency was improved by 15%.",
    "The integration of the module completed successfully.",
    "User engagement metrics have plateaued this quarter.",
    "The archival process ensures data integrity over time."
]

def generate_huge_csv(filename="chaos_corpus_large.csv"):
    print(f"Generating massive dataset: {filename}...")
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Fact", "Unit", "Subtopic"])
        
        row_count = 0
        
        # 1. THE CORE DATA (Repeated to simulate a large corpus density)
        # We repeat the definitions 5 times to ensure the vector space is crowded
        for _ in range(5):
            for word, domains in AMBIGUOUS_DATA.items():
                for domain, facts in domains.items():
                    for fact in facts:
                        writer.writerow([fact, domain, word])
                        row_count += 1
        
        # 2. THE CONFUSION LAYER (Slight variations)
        # We take the words and scramble them into sentences that look relevant but are noise
        for word in AMBIGUOUS_DATA.keys():
            for _ in range(20):
                sentence = f"The relevance of the {word} in this context is subject to debate regarding its classification."
                writer.writerow([sentence, "Academic", "Confusion"])
                row_count += 1

        # 3. PURE NOISE (Filler to force the retriever to search harder)
        for _ in range(200):
            writer.writerow([random.choice(NOISE_SENTENCES), "Noise", "General"])
            row_count += 1
            
    print(f"Done. Generated {row_count} rows.")
    print("Run ingestion: uv run elerag_improved.py ingest chaos_corpus_large.csv")

if __name__ == "__main__":
    generate_huge_csv()
