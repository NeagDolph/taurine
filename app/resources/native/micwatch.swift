import CoreAudio
import Darwin
import Foundation

final class MicWatcher {
    private let queue = DispatchQueue.main
    private let systemObject = AudioObjectID(kAudioObjectSystemObject)
    private var device = AudioDeviceID(kAudioObjectUnknown)
    private var lastActive: Bool?

    private lazy var defaultDeviceListener: AudioObjectPropertyListenerBlock = {
        [weak self] _, _ in
        self?.subscribeToDefaultInput()
    }

    private lazy var runningListener: AudioObjectPropertyListenerBlock = {
        [weak self] _, _ in
        self?.emitCurrentState()
    }

    private var defaultDeviceAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )

    private var runningAddress = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceIsRunningSomewhere,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )

    func start() {
        AudioObjectAddPropertyListenerBlock(
            systemObject,
            &defaultDeviceAddress,
            queue,
            defaultDeviceListener
        )
        subscribeToDefaultInput()
    }

    func stop() {
        AudioObjectRemovePropertyListenerBlock(
            systemObject,
            &defaultDeviceAddress,
            queue,
            defaultDeviceListener
        )
        unsubscribeFromDevice()
    }

    private func subscribeToDefaultInput() {
        unsubscribeFromDevice()

        var nextDevice = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(
            systemObject,
            &defaultDeviceAddress,
            0,
            nil,
            &size,
            &nextDevice
        ) == noErr else {
            emit(false)
            return
        }

        device = nextDevice
        AudioObjectAddPropertyListenerBlock(device, &runningAddress, queue, runningListener)
        emitCurrentState()
    }

    private func unsubscribeFromDevice() {
        guard device != kAudioObjectUnknown else { return }
        AudioObjectRemovePropertyListenerBlock(device, &runningAddress, queue, runningListener)
        device = AudioDeviceID(kAudioObjectUnknown)
    }

    private func emitCurrentState() {
        guard device != kAudioObjectUnknown else {
            emit(false)
            return
        }

        var running: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        let status = AudioObjectGetPropertyData(
            device,
            &runningAddress,
            0,
            nil,
            &size,
            &running
        )
        emit(status == noErr && running != 0)
    }

    private func emit(_ active: Bool) {
        guard active != lastActive else { return }
        lastActive = active
        print("{\"active\":\(active ? "true" : "false")}")
        fflush(stdout)
    }
}

let watcher = MicWatcher()
watcher.start()

signal(SIGTERM, SIG_IGN)
let termination = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
termination.setEventHandler {
    watcher.stop()
    exit(EXIT_SUCCESS)
}
termination.resume()
dispatchMain()
