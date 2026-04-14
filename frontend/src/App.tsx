import { NavLink, Route, Routes } from "react-router-dom";
import ChatView from "./components/ChatView";
import CollectionsView from "./components/CollectionsView";

function App() {
  return (
    <div className="flex h-screen flex-col bg-gray-900 text-gray-100">
      {/* Top navigation bar */}
      <nav className="flex items-center gap-6 border-b border-gray-700 bg-gray-800 px-6 py-3">
        <span className="text-lg font-semibold text-white">
          LangGraph RAG Chatbot
        </span>
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `text-sm font-medium transition-colors ${
              isActive
                ? "text-indigo-400"
                : "text-gray-400 hover:text-gray-200"
            }`
          }
        >
          Chat
        </NavLink>
        <NavLink
          to="/collections"
          className={({ isActive }) =>
            `text-sm font-medium transition-colors ${
              isActive
                ? "text-indigo-400"
                : "text-gray-400 hover:text-gray-200"
            }`
          }
        >
          Collections
        </NavLink>
      </nav>

      {/* Page content */}
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatView />} />
          <Route path="/collections" element={<CollectionsView />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
