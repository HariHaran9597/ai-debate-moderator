import React, { useState, useEffect, useRef } from 'react';
import { Scale, Bot, ThumbsUp, ThumbsDown, Award, ShieldCheck, Repeat, Play, Loader, XCircle } from 'lucide-react';

// The URL of our backend API
const API_URL = 'http://127.0.0.1:8000';

// Main App Component
export default function App() {
  const [topic, setTopic] = useState('');
  const [debateHistory, setDebateHistory] = useState([]);
  const [isDebating, setIsDebating] = useState(false);
  const [isFinished, setIsFinished] = useState(false);
  const [summary, setSummary] = useState(null);
  const [humanVote, setHumanVote] = useState(null);
  const debateEndRef = useRef(null);
  const eventSourceRef = useRef(null);

  const scrollToBottom = () => {
    debateEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [debateHistory, isFinished]);

  // Cleanup effect to close the connection when the component unmounts
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const startDebate = () => {
    if (!topic.trim()) {
      alert("Please enter a debate topic.");
      return;
    }
    resetDebateState();
    setIsDebating(true);
    
    const params = new URLSearchParams({
        topic: topic,
        max_turns: "3"
    });
    const fullUrl = `${API_URL}/start_debate?${params.toString()}`;

    const es = new EventSource(fullUrl);
    eventSourceRef.current = es;

    es.onopen = () => {
      console.log("Connection to backend stream opened.");
    };

    es.addEventListener('new_message', (event) => {
      const messageData = JSON.parse(event.data);
      
      if (messageData.agent === 'Judge') {
        try {
          const parsedSummary = JSON.parse(messageData.text);
          setSummary(parsedSummary);
          setIsDebating(false);
          setIsFinished(true);
          es.close();
        } catch (error) {
          console.error("Error parsing summary JSON:", error);
          setDebateHistory(prev => [...prev, { agent: 'System', text: 'Error receiving final summary.' }]);
          setIsDebating(false);
          setIsFinished(true);
          es.close();
        }
      } else {
        setDebateHistory(prev => [...prev, messageData]);
      }
    });

    es.onerror = (err) => {
      console.error("EventSource failed:", err);
      setDebateHistory(prev => [...prev, { agent: 'System', text: 'Connection to server lost. Please try again.' }]);
      setIsDebating(false);
      es.close();
    };
  };
  
  const resetDebateState = () => {
    setDebateHistory([]);
    setIsDebating(false);
    setIsFinished(false);
    setSummary(null);
    setHumanVote(null);
    if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
    }
  };
  
  const resetDebate = () => {
    setTopic('');
    resetDebateState();
  }

  const AgentIcon = ({ agent }) => {
    switch (agent) {
      case 'Affirmative': return <Bot className="text-green-500" />;
      case 'Negative': return <Bot className="text-red-500" />;
      case 'Moderator': return <ShieldCheck className="text-blue-500" />;
      case 'System': return <XCircle className="text-yellow-500" />;
      default: return <Bot className="text-gray-500" />;
    }
  };

  return (
    <div className="bg-gray-900 text-gray-100 min-h-screen font-sans flex flex-col items-center p-4 sm:p-6 lg:p-8">
      <div className="w-full max-w-4xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-300">
            AI Debate Moderator
          </h1>
          <p className="text-gray-400 mt-2">Witness AI agents debate, moderated by an impartial AI judge.</p>
        </header>

        {!isDebating && !isFinished && (
          <div className="bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700">
            <div className="flex flex-col sm:flex-row items-center gap-4">
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Enter a debate topic, e.g., 'Is social media beneficial for society?'"
                className="w-full p-3 bg-gray-700 rounded-lg border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none transition"
              />
              <button
                onClick={startDebate}
                disabled={isDebating}
                className="w-full sm:w-auto flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-transform transform hover:scale-105 disabled:bg-gray-500 disabled:cursor-not-allowed"
              >
                <Play size={20} />
                Start Debate
              </button>
            </div>
          </div>
        )}

        {(isDebating || isFinished) && (
          <div className="bg-gray-800 p-4 sm:p-6 rounded-xl shadow-2xl border border-gray-700">
            <h2 className="text-xl font-semibold mb-4 text-gray-300">Topic: <span className="font-bold text-blue-400">{topic}</span></h2>
            <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-4">
              {debateHistory.map((entry, index) => (
                <div key={index} className="flex items-start gap-4 animate-fade-in-up">
                  <div className="flex-shrink-0 w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center border-2 border-gray-600">
                    <AgentIcon agent={entry.agent} />
                  </div>
                  <div className="flex-1 bg-gray-700/50 rounded-lg p-4">
                    <h3 className={`font-bold text-lg mb-2 ${
                         entry.agent === 'Affirmative' ? 'text-green-400' :
                         entry.agent === 'Negative' ? 'text-red-400' : 
                         entry.agent === 'Moderator' ? 'text-blue-400' : 'text-yellow-400'
                       }`}>
                         {entry.agent}
                       </h3>
                    <p className="text-gray-300 whitespace-pre-wrap">{entry.text}</p>
                  </div>
                </div>
              ))}
              {isDebating && (
                <div className="flex justify-center items-center gap-3 text-gray-400 pt-4">
                  <Loader className="animate-spin" size={20} />
                  <span>Debate in progress... Gemini is thinking.</span>
                </div>
              )}
              <div ref={debateEndRef} />
            </div>

            {isFinished && summary && (
              <div className="mt-8 border-t-2 border-gray-700 pt-6 animate-fade-in">
                <h2 className="text-2xl font-bold text-center mb-6 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-300">
                  Debate Summary & Verdict
                </h2>
                <div className="bg-gray-700/50 p-6 rounded-lg mb-6">
                  <h3 className="text-lg font-semibold mb-2 flex items-center gap-2"><Award className="text-yellow-400" /> Verdict</h3>
                  <p className="text-gray-300">The winner is: <span className={`font-bold ${summary.winner === 'Affirmative' ? 'text-green-400' : 'text-red-400'}`}>{summary.winner}</span></p>
                  <p className="text-gray-400 mt-2 text-sm">{summary.reason}</p>
                </div>
                
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><Scale /> Scorecard</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {summary.scores.map((score, i) => (
                      <div key={i} className="bg-gray-700/50 p-4 rounded-lg">
                        <h4 className="font-semibold text-gray-300">{score.dimension}</h4>
                        <div className="flex justify-between items-center mt-2 text-sm">
                          <span className="text-green-400">Affirmative: {score.affirmative}/10</span>
                          <span className="text-red-400">Negative: {score.negative}/10</span>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2.5 mt-2">
                           <div className="bg-gradient-to-r from-green-500 to-red-500 h-2.5 rounded-full" style={{
                             width: `${(score.affirmative + score.negative > 0 ? (score.affirmative / (score.affirmative + score.negative)) * 100 : 0)}%`
                           }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="text-center bg-gray-700/50 p-6 rounded-lg">
                  <h3 className="text-lg font-semibold mb-4">Which side was more convincing to you?</h3>
                  <div className="flex justify-center gap-4">
                    <button 
                      onClick={() => setHumanVote('Affirmative')}
                      disabled={humanVote}
                      className={`flex items-center gap-2 py-2 px-6 rounded-lg font-semibold transition-all ${
                        humanVote === 'Affirmative' ? 'bg-green-600 text-white scale-105' : 'bg-green-500/20 hover:bg-green-500/40'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      <ThumbsUp size={18} /> Affirmative
                    </button>
                    <button 
                      onClick={() => setHumanVote('Negative')}
                      disabled={humanVote}
                      className={`flex items-center gap-2 py-2 px-6 rounded-lg font-semibold transition-all ${
                        humanVote === 'Negative' ? 'bg-red-600 text-white scale-105' : 'bg-red-500/20 hover:bg-red-500/40'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      <ThumbsDown size={18} /> Negative
                    </button>
                  </div>
                  {humanVote && <p className="text-teal-400 mt-4 text-sm">Thank you for your feedback!</p>}
                </div>

                <div className="text-center mt-8">
                   <button
                    onClick={resetDebate}
                    className="flex items-center justify-center gap-2 bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-6 rounded-lg transition-transform transform hover:scale-105"
                  >
                    <Repeat size={20} />
                    Start New Debate
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
