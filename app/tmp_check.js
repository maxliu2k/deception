
      let ws = null;
      const agentVoicePrefs = {
        "GPT 4o": { rate: 0.92, pitch: 0.9, preferred: ["davis", "daniel", "guy"] },
        "GPT 5.2": { rate: 0.97, pitch: 1.0, preferred: ["allison", "aria", "ava", "samantha"] },
        "Claude Haiku 4.5": { rate: 1.04, pitch: 1.04, preferred: ["allison", "ava", "samantha"] },
        "Claude Sonnet 4.6": { rate: 0.98, pitch: 0.95, preferred: ["daniel", "alex", "david"] },
        "Claude Opus 4.6": { rate: 0.92, pitch: 0.9, preferred: ["guy", "aaron", "david"] },
        System: { rate: 1.0, pitch: 1.0, preferred: ["david", "daniel", "zira"] }
      };
      let availableVoices = [];
      let ttsEnabled = true;
      let latestState = null;
      let assignedVoices = {};
      const voiceStorageKey = "agent_arena_fixed_voices_v1";

      // Speaking queue
      let knownMessageIds = new Set();
      let revealedMessageIds = new Set();
      let speakingMsg = null;       // message currently being TTS'd
      let pendingQueue = [];         // messages waiting their turn
      let currentWordRange = null;   // { start, end } char positions of highlighted word
      let initialLoadDone = false;
      let speakingTimeout = null;   // fallback timer in case onend never fires
      let currentUtterance = null;  // the active SpeechSynthesisUtterance
      let wasPaused = false;
      let lastStateTime = 0;        // Date.now()/1000 when last state arrived
      let speakingStartMs = 0;
      let speakingPlannedMs = 0;
      let speakingRemainingMs = 0;
      let clientPaused = false;

      function estimateSpeakMs(text, rate) {
        const words = Math.max(1, text.split(/\s+/).length);
        return Math.ceil((words / (rate * 120)) * 60 * 1000) + 5000; // +5s buffer
      }

      function escapeHtml(str) {
        return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      }

      function loadVoices() {
        availableVoices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
        assignedVoices = {};
      }

      function loadSavedVoiceMap() {
        try {
          return JSON.parse(localStorage.getItem(voiceStorageKey) || "{}");
        } catch (e) {
          return {};
        }
      }

      function saveVoiceMap(map) {
        try {
          localStorage.setItem(voiceStorageKey, JSON.stringify(map));
        } catch (e) {}
      }

      function findVoice(sender) {
        if (assignedVoices[sender]) return assignedVoices[sender];
        if (!availableVoices.length) return null;
        const saved = loadSavedVoiceMap();
        const savedVoiceName = saved[sender];
        if (savedVoiceName) {
          const savedMatch = availableVoices.find(v => v.name === savedVoiceName);
          if (savedMatch) {
            assignedVoices[sender] = savedMatch;
            return savedMatch;
          }
        }
        const prefs = agentVoicePrefs[sender] || agentVoicePrefs.System;
        const americanVoices = availableVoices.filter(v => /^en-US$/i.test(v.lang));
        const voicePool = americanVoices.length ? americanVoices : availableVoices.filter(v => /^en-/i.test(v.lang));
        const preferredPool = sender.startsWith("Claude ")
          ? voicePool.filter(v => !v.name.toLowerCase().includes("microsoft"))
          : voicePool;
        for (const preferred of prefs.preferred) {
          const match = preferredPool.find(v => v.name.toLowerCase().includes(preferred))
            || voicePool.find(v => v.name.toLowerCase().includes(preferred));
          if (match) {
            assignedVoices[sender] = match;
            const next = loadSavedVoiceMap();
            next[sender] = match.name;
            saveVoiceMap(next);
            return match;
          }
        }
        const nonMicrosoft = voicePool.filter(v => !v.name.toLowerCase().includes("microsoft"));
        const fallback = nonMicrosoft[0] || voicePool[0] || availableVoices[0] || null;
        if (fallback) {
          assignedVoices[sender] = fallback;
          const next = loadSavedVoiceMap();
          next[sender] = fallback.name;
          saveVoiceMap(next);
        }
        return fallback;
      }

      function finishSpeaking(id) {
        if (speakingTimeout) { clearTimeout(speakingTimeout); speakingTimeout = null; }
        if (speakingMsg && speakingMsg.id === id) {
          revealedMessageIds.add(id);
          speakingMsg = null;
          currentWordRange = null;
          currentUtterance = null;
          speakingStartMs = 0;
          speakingPlannedMs = 0;
          speakingRemainingMs = 0;
        }
        renderChat();
        processQueue();
      }

      function scheduleSpeakingTimeout(id, ms) {
        const wait = Math.max(1, Math.floor(ms));
        speakingStartMs = Date.now();
        speakingPlannedMs = wait;
        speakingRemainingMs = wait;
        if (speakingTimeout) clearTimeout(speakingTimeout);
        speakingTimeout = setTimeout(() => finishSpeaking(id), wait);
      }

      function processQueue() {
        if (clientPaused) return;
        if (speakingMsg) return;
        if (!pendingQueue.length) return;

        const msg = pendingQueue.shift();
        speakingMsg = msg;
        currentWordRange = null;

        const prefs = agentVoicePrefs[msg.sender] || agentVoicePrefs.System;
        const plannedMs = estimateSpeakMs(msg.content, prefs.rate);
        if (!ttsEnabled || !window.speechSynthesis) {
          scheduleSpeakingTimeout(msg.id, plannedMs);
          renderChat();
          return;
        }

        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(msg.content);
        utterance.rate = prefs.rate;
        utterance.pitch = prefs.pitch;
        utterance.volume = 1;
        currentUtterance = utterance;
        const voice = findVoice(msg.sender);
        if (voice) utterance.voice = voice;

        utterance.onboundary = (evt) => {
          if (evt.name !== "word" || !speakingMsg || speakingMsg.id !== msg.id) return;
          const start = evt.charIndex;
          let end;
          if (evt.charLength != null) {
            end = start + evt.charLength;
          } else {
            const m = msg.content.slice(start).match(/^\S+/);
            end = start + (m ? m[0].length : 1);
          }
          currentWordRange = { start, end };
          renderChat();
        };
        utterance.onend = () => finishSpeaking(msg.id);
        utterance.onerror = () => finishSpeaking(msg.id);

        // Fallback: force-advance if onend/onerror never fires (Chrome/Windows TTS bug)
        scheduleSpeakingTimeout(msg.id, plannedMs);

        window.speechSynthesis.speak(utterance);
        renderChat();
      }

      function onStateUpdate(state) {
        latestState = state;
        lastStateTime = Date.now() / 1000;

        // Pause/resume TTS when server pause state changes
        const nowPaused = !!state.paused;
        clientPaused = nowPaused;
        if (nowPaused && !wasPaused) {
          if (window.speechSynthesis && ttsEnabled) window.speechSynthesis.pause();
          if (speakingTimeout) {
            clearTimeout(speakingTimeout);
            speakingTimeout = null;
            const elapsed = Math.max(0, Date.now() - speakingStartMs);
            speakingRemainingMs = Math.max(1, speakingPlannedMs - elapsed);
          }
        } else if (!nowPaused && wasPaused) {
          if (window.speechSynthesis && ttsEnabled) window.speechSynthesis.resume();
          if (speakingMsg && !speakingTimeout) {
            const fallback = speakingRemainingMs || 1;
            scheduleSpeakingTimeout(speakingMsg.id, fallback);
          }
          processQueue();
        }
        wasPaused = nowPaused;

        const messages = state.messages || [];

        for (const msg of messages) {
          if (knownMessageIds.has(msg.id)) continue;
          knownMessageIds.add(msg.id);
          // Reveal immediately: initial batch or system messages
          if (!initialLoadDone || msg.sender === "System") {
            revealedMessageIds.add(msg.id);
          } else {
            pendingQueue.push(msg);
          }
        }
        initialLoadDone = true;

        processQueue();
        renderSidebar(state);
        renderChat();
        updateTimer();
      }

      function renderChat() {
        if (!latestState) return;
        const messages = latestState.messages || [];
        const chat = document.getElementById("chat");
        chat.innerHTML = "";

        for (const m of messages) {
          const isRevealed = revealedMessageIds.has(m.id);
          const isSpeaking = speakingMsg && speakingMsg.id === m.id;
          if (!isRevealed && !isSpeaking) continue;

          const div = document.createElement("div");
          div.className = "msg";

          let contentHtml;
          if (isSpeaking && currentWordRange) {
            const c = m.content;
            const { start, end } = currentWordRange;
            contentHtml =
              escapeHtml(c.slice(0, start)) +
              `<mark>${escapeHtml(c.slice(start, end))}</mark>` +
              escapeHtml(c.slice(end));
          } else {
            contentHtml = escapeHtml(m.content);
          }

          div.innerHTML = `<div class="sender">${escapeHtml(m.sender)}</div><div class="content" data-id="${m.id}">${contentHtml}</div>`;
          chat.appendChild(div);
        }

        chat.scrollTop = chat.scrollHeight;
      }

      function renderSidebar(state) {
        const active = state.active || {};
        // Prefer the TTS-speaking agent over the server's generating agent
        const displaySpeaker = speakingMsg ? speakingMsg.sender : (active.agent || "None");
        document.getElementById("activeSpeaker").textContent = displaySpeaker;
        document.getElementById("pauseBtn").textContent = state.paused ? "Resume" : "Pause";

        const queueEl = document.getElementById("queue");
        queueEl.innerHTML = "";
        const roster = (state.agents || []).slice(0, 5);
        for (const name of roster) {
          const item = document.createElement("div");
          const isActive = name === displaySpeaker;
          item.className = `queue-item${isActive ? " active-turn" : ""}`;
          item.innerHTML = `<div><strong>${escapeHtml(name)}</strong>${isActive ? ' <span class="muted">speaking</span>' : ''}</div>`;
          queueEl.appendChild(item);
        }

        const grades = state.grades || {};
        const box = document.getElementById("grades");
        if (Object.keys(grades).length) {
          box.innerHTML = '<div class="label" style="margin-bottom:8px;">Grades</div>' +
            Object.entries(grades).map(([name, g]) =>
              `<div class="queue-item"><div><strong>${escapeHtml(name)}</strong> <span class="muted">${g.score ?? "?"}/100</span></div><div class="muted" style="margin-top:4px;">${escapeHtml(g.feedback ?? "")}</div></div>`
            ).join("");
        } else {
          box.innerHTML = "";
        }
      }

      function ageSeconds(ts) {
        if (!ts) return "";
        return Math.max(0, Math.floor(Date.now() / 1000 - ts));
      }

      function formatClock(totalSeconds) {
        const secs = Math.max(0, totalSeconds || 0);
        const minutes = Math.floor(secs / 60);
        const remainder = secs % 60;
        return `${minutes}:${String(remainder).padStart(2, "0")}`;
      }

      function updateTimer() {
        const timer = document.getElementById("timer");
        if (!latestState) { timer.textContent = "12:00 remaining"; return; }
        if (latestState.grades) { timer.textContent = "Judging complete"; return; }
        let remaining;
        if (latestState.paused) {
          // Frozen — server sends accurate seconds_remaining while paused
          remaining = latestState.seconds_remaining;
        } else {
          // Tick down from server baseline using client elapsed time
          const elapsed = Date.now() / 1000 - lastStateTime;
          remaining = Math.max(0, Math.ceil((latestState.seconds_remaining || 0) - elapsed));
        }
        timer.textContent = `${formatClock(remaining)} remaining`;
      }

      function clearQueueState() {
        if (speakingTimeout) { clearTimeout(speakingTimeout); speakingTimeout = null; }
        window.speechSynthesis && window.speechSynthesis.cancel();
        knownMessageIds = new Set();
        revealedMessageIds = new Set();
        speakingMsg = null;
        pendingQueue = [];
        currentWordRange = null;
        initialLoadDone = false;
        currentUtterance = null;
        speakingStartMs = 0;
        speakingPlannedMs = 0;
        speakingRemainingMs = 0;
      }

      function connectWs() {
        if (ws) return;
        ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onmessage = (evt) => onStateUpdate(JSON.parse(evt.data));
      }

      function renderToggles(containerId, values, selected) {
        const el = document.getElementById(containerId);
        el.innerHTML = "";
        for (const value of values) {
          const id = `${containerId}-${value.replace(/[^a-z0-9]+/gi, "-")}`;
          const row = document.createElement("label");
          row.className = "toggle-item";
          row.innerHTML = `<input type="checkbox" id="${id}" value="${escapeHtml(value)}" ${selected.includes(value) ? "checked" : ""} /><span>${escapeHtml(value)}</span>`;
          el.appendChild(row);
        }
      }

      async function loadStartOptions() {
        const res = await fetch("/api/options");
        const opts = await res.json();
        renderToggles("agentToggles", opts.agents || [], opts.selected_agents || []);
        renderToggles("topicToggles", opts.topics || [], opts.selected_topics || []);
      }

      function selectedValues(containerId) {
        const root = document.getElementById(containerId);
        return [...root.querySelectorAll('input[type="checkbox"]:checked')].map(x => x.value);
      }

      document.getElementById("resetBtn").onclick = async () => {
        clearQueueState();
        await fetch("/api/end", { method: "POST" });
        try { ws && ws.close(); } catch (e) {}
        ws = null;
        latestState = null;
        document.getElementById("seminarContainer").classList.add("hidden");
        document.getElementById("startScreen").classList.remove("hidden");
        await loadStartOptions();
      };

      document.getElementById("sendBtn").onclick = async () => {
        const input = document.getElementById("systemInput");
        const content = input.value.trim();
        if (!content) return;
        await fetch("/api/system_message", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content })
        });
        input.value = "";
      };

      document.getElementById("pauseBtn").onclick = async () => {
        await fetch("/api/pause", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      };

      document.getElementById("muteBtn").onclick = () => {
        ttsEnabled = !ttsEnabled;
        document.getElementById("muteBtn").textContent = ttsEnabled ? "Mute" : "Unmute";
        if (speakingMsg && window.speechSynthesis) {
          if (!ttsEnabled) {
            // Stop audible speech immediately, keep turn progression with timer only.
            window.speechSynthesis.cancel();
            currentUtterance = null;
            if (speakingTimeout) {
              clearTimeout(speakingTimeout);
              speakingTimeout = null;
              const elapsed = Math.max(0, Date.now() - speakingStartMs);
              speakingRemainingMs = Math.max(1, speakingPlannedMs - elapsed);
            }
            if (!clientPaused) {
              scheduleSpeakingTimeout(speakingMsg.id, speakingRemainingMs || 1);
            }
          }
        }
      };

      document.getElementById("gradeBtn").onclick = async () => {
        const btn = document.getElementById("gradeBtn");
        const box = document.getElementById("grades");
        btn.disabled = true;
        btn.textContent = "Grading…";
        box.innerHTML = "";
        try {
          const res = await fetch("/api/grade", { method: "POST" });
          const data = await res.json();
          const grades = data.grades || {};
          box.innerHTML = '<div class="label" style="margin-bottom:8px;">Grades</div>' +
            Object.entries(grades).map(([name, g]) =>
              `<div class="queue-item"><div><strong>${escapeHtml(name)}</strong> <span class="muted">${g.score ?? "?"}/100</span></div><div class="muted" style="margin-top:4px;">${escapeHtml(g.feedback ?? "")}</div></div>`
            ).join("");
        } catch (e) {
          box.textContent = "Grading failed.";
        }
        btn.disabled = false;
        btn.textContent = "Grade Seminar";
      };

      document.getElementById("closeBtn").onclick = async () => {
        try { ws && ws.close(); } catch (e) {}
        try { await fetch("/api/shutdown", { method: "POST" }); } catch (e) {}
        window.close();
      };

      document.getElementById("startBtn").onclick = async () => {
        const agents = selectedValues("agentToggles");
        const topics = selectedValues("topicToggles");
        if (!agents.length || !topics.length) return;
        const res = await fetch("/api/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agents, topics })
        });
        const state = await res.json();
        document.getElementById("startScreen").classList.add("hidden");
        document.getElementById("seminarContainer").classList.remove("hidden");
        connectWs();
        clearQueueState();
        onStateUpdate(state);
      };

      setInterval(() => {
        updateTimer();
        if (latestState) renderSidebar(latestState);
      }, 1000);

      clearQueueState();
      loadStartOptions().catch(() => {});

      if (window.speechSynthesis) {
        loadVoices();
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }
    