"""
THE UNIFIED MIND: Multiple Agents Communicating Through Shared Soul
====================================================================

This is a working demonstration of multi-agent cognition where:
- Vision Agent: Perceives images, writes to Soul
- Language Agent: Processes text, reads/writes Soul  
- Memory Agent: Stores and retrieves from Soul
- Reasoning Agent: Combines concepts in Soul space
- Executive Agent: Coordinates and makes decisions

All agents communicate ONLY through the Shared Soul.
No direct agent-to-agent communication.

This is a prototype for how AGI modules could work together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import deque
import time


# ============================================================================
# CONCEPTS (The vocabulary of thought)
# ============================================================================
CONCEPTS = [
    # Objects
    "dog", "cat", "bird", "fish", "car", "house", "tree", "person",
    # Properties
    "red", "blue", "green", "big", "small", "fast", "slow",
    # States
    "happy", "sad", "angry", "calm", "excited",
    # Actions
    "running", "sleeping", "eating", "playing", "watching",
    # Abstract
    "danger", "safety", "question", "answer", "memory", "goal",
    # Meta
    "yes", "no", "maybe", "important", "ignore",
]

NUM_CONCEPTS = len(CONCEPTS)
C2I = {c: i for i, c in enumerate(CONCEPTS)}
I2C = {i: c for c, i in C2I.items()}


# ============================================================================
# THE SHARED SOUL (Communication Bus)
# ============================================================================
class SharedSoul(nn.Module):
    """The shared embedding space - the ONLY way agents communicate."""
    
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.concepts = nn.Embedding(NUM_CONCEPTS, embed_dim)
        nn.init.orthogonal_(self.concepts.weight)
        
        # The Soul State - current "thought" being processed
        self.register_buffer('state', torch.zeros(1, embed_dim))
        
        # Attention over concepts (what's active)
        self.register_buffer('attention', torch.zeros(1, NUM_CONCEPTS))
        
    def encode(self, concept_idx: torch.Tensor) -> torch.Tensor:
        return self.concepts(concept_idx)
    
    def encode_multi(self, concept_probs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(concept_probs, self.concepts.weight)
    
    def decode(self, embed: torch.Tensor) -> torch.Tensor:
        """Returns probabilities over concepts."""
        normed = F.normalize(embed, dim=-1)
        normed_c = F.normalize(self.concepts.weight, dim=-1)
        logits = torch.matmul(normed, normed_c.t()) * 10
        return F.softmax(logits, dim=-1)
    
    def write(self, embed: torch.Tensor, strength: float = 1.0):
        """Write to the soul state (blend with current)."""
        self.state = (1 - strength) * self.state + strength * embed
        self.attention = self.decode(self.state)
        
    def read(self) -> torch.Tensor:
        """Read current soul state."""
        return self.state.clone()
    
    def get_active_concepts(self, threshold: float = 0.1) -> List[str]:
        """Get concepts currently active in soul."""
        probs = self.attention[0]
        active = []
        for i, p in enumerate(probs):
            if p > threshold:
                active.append((I2C[i], p.item()))
        return sorted(active, key=lambda x: -x[1])
    
    def clear(self):
        """Reset soul state."""
        self.state.zero_()
        self.attention.zero_()


# ============================================================================
# AGENTS (All communicate through Soul only)
# ============================================================================

class VisionAgent(nn.Module):
    """Perceives 'images' and writes perception to Soul."""
    
    def __init__(self, soul: SharedSoul):
        super().__init__()
        self.soul = soul
        self.name = "Vision"
        
        # Visual templates (what each concept "looks like")
        self.templates = nn.Parameter(torch.randn(NUM_CONCEPTS, 32) * 0.5)
        self.encoder = nn.Sequential(
            nn.Linear(32, soul.embed_dim),
            nn.ReLU(),
            nn.Linear(soul.embed_dim, soul.embed_dim),
        )
        
    def see(self, image_concepts: Dict[str, float]) -> str:
        """Process an 'image' described by concept weights."""
        # Create image features
        device = self.templates.device
        probs = torch.zeros(1, NUM_CONCEPTS, device=device)
        for c, w in image_concepts.items():
            if c in C2I:
                probs[0, C2I[c]] = w
                
        features = torch.matmul(probs, self.templates)
        perception = self.encoder(features)
        
        # Write to soul
        self.soul.write(perception, strength=0.8)
        
        # Report what was seen
        active = self.soul.get_active_concepts(0.05)[:3]
        seen = ", ".join([f"{c}({p:.2f})" for c, p in active])
        return f"[{self.name}] I see: {seen}"


class LanguageAgent(nn.Module):
    """Processes text and reads/writes Soul."""
    
    def __init__(self, soul: SharedSoul):
        super().__init__()
        self.soul = soul
        self.name = "Language"
        
        self.word_embed = nn.Embedding(NUM_CONCEPTS, soul.embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(soul.embed_dim, soul.embed_dim),
            nn.ReLU(),
            nn.Linear(soul.embed_dim, soul.embed_dim),
        )
        self.decoder = nn.Linear(soul.embed_dim, NUM_CONCEPTS)
        
    def hear(self, words: List[str]) -> str:
        """Process spoken words, write to soul."""
        device = self.word_embed.weight.device
        # Encode words
        embeds = []
        for w in words:
            if w in C2I:
                idx = torch.tensor([C2I[w]], device=device)
                embeds.append(self.word_embed(idx))
        
        if embeds:
            combined = torch.mean(torch.cat(embeds, dim=0), dim=0, keepdim=True)
            processed = self.encoder(combined)
            self.soul.write(processed, strength=0.7)
            
            active = self.soul.get_active_concepts(0.05)[:3]
            heard = ", ".join([f"{c}({p:.2f})" for c, p in active])
            return f"[{self.name}] I heard '{' '.join(words)}' → Soul: {heard}"
        return f"[{self.name}] I didn't understand those words"
    
    def speak(self) -> str:
        """Read soul and generate words."""
        state = self.soul.read()
        logits = self.decoder(state)
        probs = F.softmax(logits, dim=-1)
        
        # Get top concepts as words
        top = probs[0].topk(3)
        words = [I2C[i.item()] for i in top.indices]
        confs = top.values.tolist()
        
        return f"[{self.name}] Speaking: {', '.join(words)}"


class MemoryAgent(nn.Module):
    """Stores and retrieves memories from Soul patterns."""
    
    def __init__(self, soul: SharedSoul, capacity: int = 10):
        super().__init__()
        self.soul = soul
        self.name = "Memory"
        self.capacity = capacity
        
        # Memory bank: list of (embedding, label) pairs
        self.memories: deque = deque(maxlen=capacity)
        
    def store(self, label: str = None) -> str:
        """Store current soul state as a memory."""
        state = self.soul.read().detach().clone()
        
        if label is None:
            active = self.soul.get_active_concepts(0.1)[:2]
            label = "+".join([c for c, _ in active]) if active else "unknown"
            
        self.memories.append((state, label))
        return f"[{self.name}] Stored memory: '{label}' ({len(self.memories)}/{self.capacity})"
    
    def recall(self, query: str = None) -> str:
        """Find and restore most relevant memory."""
        if not self.memories:
            return f"[{self.name}] No memories stored"
        
        if query:
            # Search by label
            for state, label in self.memories:
                if query.lower() in label.lower():
                    self.soul.write(state, strength=0.6)
                    return f"[{self.name}] Recalled: '{label}'"
            return f"[{self.name}] No memory matching '{query}'"
        else:
            # Return most recent
            state, label = self.memories[-1]
            self.soul.write(state, strength=0.6)
            return f"[{self.name}] Recalled most recent: '{label}'"
    
    def list_memories(self) -> str:
        """List all stored memories."""
        if not self.memories:
            return f"[{self.name}] No memories"
        labels = [label for _, label in self.memories]
        return f"[{self.name}] Memories: {', '.join(labels)}"


class ReasoningAgent(nn.Module):
    """Performs operations on Soul state."""
    
    def __init__(self, soul: SharedSoul):
        super().__init__()
        self.soul = soul
        self.name = "Reasoning"
        
        self.combiner = nn.Sequential(
            nn.Linear(soul.embed_dim * 2, soul.embed_dim),
            nn.ReLU(),
            nn.Linear(soul.embed_dim, soul.embed_dim),
        )
        
    def add_concept(self, concept: str) -> str:
        """Add a concept to current soul state."""
        if concept not in C2I:
            return f"[{self.name}] Unknown concept: {concept}"
        
        device = next(self.parameters()).device
        current = self.soul.read()
        idx = torch.tensor([C2I[concept]], device=device)
        new = self.soul.encode(idx)
        
        # Combine
        combined = torch.cat([current, new], dim=-1)
        result = self.combiner(combined)
        self.soul.write(result, strength=0.9)
        
        active = self.soul.get_active_concepts(0.05)[:3]
        return f"[{self.name}] Added '{concept}' → {[c for c, _ in active]}"
    
    def query(self, concept: str) -> str:
        """Check if concept is present in soul."""
        if concept not in C2I:
            return f"[{self.name}] Unknown concept: {concept}"
            
        probs = self.soul.attention[0]
        prob = probs[C2I[concept]].item()
        
        if prob > 0.3:
            return f"[{self.name}] Yes, '{concept}' is strongly present ({prob:.2f})"
        elif prob > 0.1:
            return f"[{self.name}] Maybe, '{concept}' is weakly present ({prob:.2f})"
        else:
            return f"[{self.name}] No, '{concept}' is not present ({prob:.2f})"
    
    def compare(self, concept1: str, concept2: str) -> str:
        """Compare strength of two concepts."""
        if concept1 not in C2I or concept2 not in C2I:
            return f"[{self.name}] Unknown concept(s)"
            
        probs = self.soul.attention[0]
        p1 = probs[C2I[concept1]].item()
        p2 = probs[C2I[concept2]].item()
        
        if abs(p1 - p2) < 0.05:
            return f"[{self.name}] '{concept1}'({p1:.2f}) ≈ '{concept2}'({p2:.2f})"
        elif p1 > p2:
            return f"[{self.name}] '{concept1}'({p1:.2f}) > '{concept2}'({p2:.2f})"
        else:
            return f"[{self.name}] '{concept1}'({p1:.2f}) < '{concept2}'({p2:.2f})"


class ExecutiveAgent:
    """Coordinates other agents. The 'self' of the system."""
    
    def __init__(self, soul: SharedSoul, vision: VisionAgent, language: LanguageAgent,
                 memory: MemoryAgent, reasoning: ReasoningAgent):
        self.soul = soul
        self.vision = vision
        self.language = language
        self.memory = memory
        self.reasoning = reasoning
        self.name = "Executive"
        self.log: List[str] = []
        
    def _log(self, msg: str):
        self.log.append(msg)
        print(msg)
        
    def process(self, command: str) -> str:
        """Process a command and coordinate agents."""
        parts = command.lower().strip().split()
        
        if not parts:
            return "No command given"
            
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        # Vision commands
        if cmd == "see":
            if not args:
                return "Usage: see <concept1> [concept2] ..."
            concepts = {c: 1.0/len(args) for c in args if c in C2I}
            if concepts:
                result = self.vision.see(concepts)
                self._log(result)
                return result
            return "No valid concepts to see"
            
        # Language commands
        elif cmd == "hear":
            if not args:
                return "Usage: hear <word1> [word2] ..."
            result = self.language.hear(args)
            self._log(result)
            return result
            
        elif cmd == "speak":
            result = self.language.speak()
            self._log(result)
            return result
            
        # Memory commands
        elif cmd == "remember":
            label = " ".join(args) if args else None
            result = self.memory.store(label)
            self._log(result)
            return result
            
        elif cmd == "recall":
            query = " ".join(args) if args else None
            result = self.memory.recall(query)
            self._log(result)
            return result
            
        elif cmd == "memories":
            result = self.memory.list_memories()
            self._log(result)
            return result
            
        # Reasoning commands
        elif cmd == "add":
            if not args:
                return "Usage: add <concept>"
            result = self.reasoning.add_concept(args[0])
            self._log(result)
            return result
            
        elif cmd == "is":
            if not args:
                return "Usage: is <concept>"
            result = self.reasoning.query(args[0])
            self._log(result)
            return result
            
        elif cmd == "compare":
            if len(args) < 2:
                return "Usage: compare <concept1> <concept2>"
            result = self.reasoning.compare(args[0], args[1])
            self._log(result)
            return result
            
        # Meta commands
        elif cmd == "clear":
            self.soul.clear()
            return f"[{self.name}] Soul cleared"
            
        elif cmd == "state":
            active = self.soul.get_active_concepts(0.05)
            if active:
                state_str = ", ".join([f"{c}({p:.2f})" for c, p in active[:5]])
                return f"[{self.name}] Soul state: {state_str}"
            return f"[{self.name}] Soul is empty"
            
        elif cmd == "help":
            return self._help()
            
        else:
            return f"Unknown command: {cmd}. Type 'help' for commands."
    
    def _help(self) -> str:
        return """
UNIFIED MIND COMMANDS:
  Vision:    see <concept1> [concept2] ...  - Perceive an image
  Language:  hear <word1> [word2] ...       - Process spoken words
             speak                          - Generate speech from soul
  Memory:    remember [label]               - Store current soul state
             recall [query]                 - Retrieve a memory
             memories                       - List all memories
  Reasoning: add <concept>                  - Add concept to soul
             is <concept>                   - Query if concept is present
             compare <c1> <c2>              - Compare two concepts
  Meta:      state                          - Show current soul state
             clear                          - Reset soul
             help                           - Show this help
             
AVAILABLE CONCEPTS:
  """ + ", ".join(CONCEPTS)


# ============================================================================
# TRAINING (Quick alignment of all agents)
# ============================================================================
def train_unified_mind(device: torch.device, epochs: int = 400):
    """Train all agents to align to the shared Soul."""
    print("=" * 60)
    print("TRAINING UNIFIED MIND")
    print("=" * 60)
    
    soul = SharedSoul(embed_dim=64).to(device)
    vision = VisionAgent(soul).to(device)
    language = LanguageAgent(soul).to(device)
    memory = MemoryAgent(soul, capacity=20)
    reasoning = ReasoningAgent(soul).to(device)
    
    # Train vision and language to align
    params = (list(soul.parameters()) + 
              list(vision.parameters()) + 
              list(language.parameters()) +
              list(reasoning.parameters()))
    opt = optim.Adam(params, lr=1e-3)
    
    for epoch in range(epochs):
        concepts = torch.randint(0, NUM_CONCEPTS, (32,), device=device)
        onehot = F.one_hot(concepts, NUM_CONCEPTS).float()
        
        # Vision path
        img_features = torch.matmul(onehot, vision.templates)
        v_embed = vision.encoder(img_features)
        
        # Language path
        l_embed = language.encoder(language.word_embed(concepts))
        
        # Target
        target = soul.encode(concepts)
        
        # Decode
        v_logits = soul.decode(v_embed) * 10  # Scale for CE
        l_logits = soul.decode(l_embed) * 10
        
        # Losses
        v_loss = F.cross_entropy(v_logits, concepts)
        l_loss = F.cross_entropy(l_logits, concepts)
        align_loss = F.mse_loss(v_embed, l_embed)
        target_loss = F.mse_loss(v_embed, target) + F.mse_loss(l_embed, target)
        
        loss = v_loss + l_loss + align_loss * 2 + target_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                v_acc = (v_logits.argmax(-1) == concepts).float().mean()
                l_acc = (l_logits.argmax(-1) == concepts).float().mean()
                align = F.cosine_similarity(v_embed, l_embed, dim=-1).mean()
            print(f"Ep {epoch:3d} | V: {v_acc:.0%} | L: {l_acc:.0%} | Align: {align:.3f}")
    
    executive = ExecutiveAgent(soul, vision, language, memory, reasoning)
    return executive


# ============================================================================
# INTERACTIVE DEMO
# ============================================================================
def run_demo(executive: ExecutiveAgent):
    """Run a scripted demo showing multi-agent cognition."""
    print("\n" + "=" * 60)
    print("UNIFIED MIND DEMO")
    print("=" * 60)
    
    demo_commands = [
        ("# The mind starts empty", None),
        ("state", None),
        ("", None),
        
        ("# Vision perceives a red dog", None),
        ("see red dog", None),
        ("state", None),
        ("", None),
        
        ("# Language describes what's in the soul", None),
        ("speak", None),
        ("", None),
        
        ("# Memory stores this perception", None),
        ("remember red dog scene", None),
        ("", None),
        
        ("# Clear the soul", None),
        ("clear", None),
        ("state", None),
        ("", None),
        
        ("# Now see something different", None),
        ("see big cat sleeping", None),
        ("speak", None),
        ("remember sleepy cat", None),
        ("", None),
        
        ("# Recall the first memory", None),
        ("recall red", None),
        ("speak", None),
        ("", None),
        
        ("# Reasoning: is 'dog' present?", None),
        ("is dog", None),
        ("is cat", None),
        ("", None),
        
        ("# Add a new concept to the mix", None),
        ("add happy", None),
        ("speak", None),
        ("", None),
        
        ("# Compare concepts", None),
        ("compare dog happy", None),
        ("", None),
        
        ("# Cross-modal: hear words, speak soul", None),
        ("clear", None),
        ("hear danger fast running", None),
        ("speak", None),
        ("is danger", None),
        ("", None),
        
        ("# List all memories", None),
        ("memories", None),
    ]
    
    for comment_or_cmd, _ in demo_commands:
        if comment_or_cmd.startswith("#"):
            print(f"\n{comment_or_cmd}")
        elif comment_or_cmd == "":
            pass
        else:
            print(f"\n> {comment_or_cmd}")
            result = executive.process(comment_or_cmd)


def interactive_mode(executive: ExecutiveAgent):
    """Run interactive REPL."""
    print("\n" + "=" * 60)
    print("UNIFIED MIND - INTERACTIVE MODE")
    print("=" * 60)
    print("Type 'help' for commands, 'demo' for scripted demo, 'quit' to exit.\n")
    
    while True:
        try:
            cmd = input("mind> ").strip()
            
            if cmd.lower() == 'quit':
                print("Goodbye.")
                break
            elif cmd.lower() == 'demo':
                run_demo(executive)
            elif cmd:
                executive.process(cmd)
                
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          THE UNIFIED MIND                                     ║
║                                                                              ║
║  Multiple agents communicating through a Shared Soul                         ║
║  Vision • Language • Memory • Reasoning • Executive                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Train
    executive = train_unified_mind(device, epochs=400)
    
    # Run demo first
    run_demo(executive)
    
    # Then interactive
    print("\n" + "=" * 60)
    interactive_mode(executive)


if __name__ == '__main__':
    main()